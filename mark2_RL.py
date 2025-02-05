import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from datasets import load_dataset
import os
from tqdm.auto import tqdm

class TextGenerationDataset(Dataset):
    def __init__(self, tokenizer, max_length=192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset(
            "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B",
            split="train[:5000]"  # Use full training split
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Combine question and answer for training
        example = self.dataset[idx]
        # print(f"{example}\n\n")
        text = f"Question: {example['instruction']}\nReasoning: {example['conversations']}\nAnswer: {example['response']}"
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            padding='max_length'
        )
        return inputs.input_ids[0], inputs.attention_mask[0]

def collate_fn(batch):
    valid_batch = [(ids, mask) for ids, mask in batch if ids.size(0) >= 2]
    if not valid_batch:
        return torch.tensor([]), torch.tensor([])
    
    input_ids, attention_masks = zip(*valid_batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    return input_ids, attention_masks

class PolicyNetwork(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        self.transformer.gradient_checkpointing_enable()  # Gradient checkpointing
        
    def forward(self, states, attention_mask=None):
        outputs = self.transformer(input_ids=states, attention_mask=attention_mask)
        return outputs.logits

class RLTextGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.policy_net = PolicyNetwork().to(device)
        self.optimizer = AdamW(self.policy_net.parameters(), lr=1e-5, weight_decay=0.01)
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed precision
        
    def prepare_dataset(self):
        return TextGenerationDataset(self.tokenizer)

    def train(self, num_episodes=1000, batch_size=16, save_every=5, grad_accum=4):
        dataset = self.prepare_dataset()
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        print(f"dataset loaded...")
        for episode in range(1, num_episodes+1):
            total_reward = 0
            progress_bar = tqdm(loader, desc=f"Episode {episode}/{num_episodes}", leave=False)
            
            for step, batch in enumerate(progress_bar):
                input_ids, attention_mask = batch
                if input_ids.size(0) == 0:
                    continue
                
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        # Add shape checks before generation
                        # print(f"Input IDs shape: {input_ids.shape}")
                        # print(f"Attention mask shape: {attention_mask.shape}")

                        generated_ids = self.policy_net.transformer.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask.squeeze(1),
                            max_length=256,  # Reduced generation length
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    rewards = self.calculate_rewards(generated_ids)
                    
                    states = generated_ids[:, :-1]
                    actions = generated_ids[:, 1:]
                    mask = (states != self.tokenizer.pad_token_id).float()
                    
                    logits = self.policy_net(states, attention_mask=mask)
                    probs = torch.softmax(logits, dim=-1)
                    selected_probs = probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                    loss = -torch.mean(torch.log(selected_probs + 1e-10) * rewards)
                    loss = loss / grad_accum  # Gradient accumulation

                # Mixed precision backward
                self.scaler.scale(loss).backward()
                
                if (step + 1) % grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_reward += rewards.sum().item()
                progress_bar.set_postfix({
                    'loss': loss.item() * grad_accum,
                    'avg_reward': total_reward / ((step + 1) * batch_size)
                })
                
                # Manual memory cleanup
                del generated_ids, rewards, states, actions, logits, probs, selected_probs
                torch.cuda.empty_cache()

            if episode % save_every == 0:
                self.save_model(f"./model/epoch_{episode}")
                print(f"\nSaved checkpoint at epoch {episode}")

            print(f"Episode {episode} | Avg Reward: {total_reward/len(loader):.2f}")

    def calculate_rewards(self, generated_ids):
        """Custom reward function - modify this for your use case"""
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
        # Custom rewards for reasoning quality
        rewards = []
        for text in decoded:
            reward = 0.0
            # Reward for containing reasoning components
            if 'reasoning:' in text.lower():
                reward += 1.0
            # Reward for answer structure
            if 'answer:' in text.lower():
                reward += 0.5
            # Penalize empty reasoning
            if 'reasoning: none' in text.lower():
                reward -= 2.0
            rewards.append(reward)
        return torch.tensor(rewards, device=self.device).unsqueeze(-1)
        
    def generate_text(self, prompt, max_length=128, temperature=0.9):
        self.policy_net.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.policy_net.transformer.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def interactive_generation(self):
        print("Interactive mode (type 'exit' to quit)")
        while True:
            prompt = input("\nEnter your prompt: ")
            if prompt.lower() == 'exit':
                break
            generated = self.generate_text(prompt)
            print(f"Generated text: {generated}")

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.policy_net.transformer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path):
        instance = cls()
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path)
        instance.policy_net = PolicyNetwork(model_name=path).to(instance.device)
        # print(instance.policy_net.parameters())
        instance.optimizer = AdamW(instance.policy_net.parameters(), lr=1e-5, weight_decay=0.01)
        
        instance.scaler = torch.cuda.amp.GradScaler() 
        return instance

if __name__ == "__main__":
    # generator = RLTextGenerator()
    
    # Load existing model instead of creating new
    generator = RLTextGenerator.load_model("./RL_model_final")  # Path to your saved model
    # Train the model with saving every 5 epochs
    print("Starting training...")
    generator.train(
        num_episodes=40,
        batch_size=24,
        save_every=5  # Save every 5 epochs
    )
    
    # Final model save
    generator.save_model("./RL_model_final")
    
    # Interactive generation
    generator.interactive_generation()