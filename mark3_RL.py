import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import os
from tqdm.auto import tqdm

# ----------------------
# Improved Dataset Class
# ----------------------
class ResumeDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Parse resume with section-aware processing
        with open(file_path, 'r') as f:
            content = f.read().split('\n\n')
            
        current_section = ""
        for part in content:
            # Check if new section header
            if ':' in part.split('\n')[0] and not part.startswith(' '):
                if current_section:  # Save previous section
                    self._add_section(current_section)
                current_section = part
            else:
                current_section += "\n" + part
                
        if current_section:  # Add final section
            self._add_section(current_section)

    def _add_section(self, section):
        """Format sections with instructional prompts"""
        header, content = section.split(':', 1)
        header = header.strip()
        content = content.strip()
        
        if header == "Personal Information":
            self.data.append(f"Generate personal information for Sainesh Patil: {content}")
        elif header == "Work Experience":
            self.data.append(f"Describe Sainesh Patil's work experience: {content}")
        elif header == "Personal Projects":
            self.data.append(f"Explain Sainesh Patil's projects: {content}")
        elif header == "Education":
            self.data.append(f"List Sainesh Patil's education details: {content}")
        elif header == "Technical Skills":
            self.data.append(f"Detail Sainesh Patil's technical skills: {content}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            padding='max_length',
            add_special_tokens=True
        )
        return inputs.input_ids[0], inputs.attention_mask[0]

# ----------------------
# Model Architecture
# ----------------------
class PolicyNetwork(nn.Module):
    def __init__(self, model_name='gpt2', tokenizer=None):
        super().__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        if tokenizer:  # Only resize if tokenizer is provided
            self.transformer.resize_token_embeddings(len(tokenizer))

    def forward(self, states, attention_mask=None):
        outputs = self.transformer(input_ids=states, attention_mask=attention_mask)
        return outputs.logits

# ----------------------
# Enhanced RL Generator
# ----------------------
class RLResumeGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pass tokenizer to PolicyNetwork
        self.policy_net = PolicyNetwork(tokenizer=self.tokenizer).to(device)
        self.optimizer = AdamW(self.policy_net.parameters(), lr=2e-5, weight_decay=0.001)
        self.scaler = torch.cuda.amp.GradScaler()

    def prepare_dataset(self):
        return ResumeDataset(self.tokenizer, 'sainesh_patil_dataset.txt')

    # ----------------------
    # Improved Training Loop
    # ----------------------
    def train(self, num_episodes=150, batch_size=4, save_every=25, grad_accum=2):
        dataset = self.prepare_dataset()
        print(f"Loaded {len(dataset)} resume sections")
        print("Sample training example:", dataset.data[0][:100] + "...")
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: (torch.stack([x[0] for x in b]), 
                                 torch.stack([x[1] for x in b]))
        )

        for episode in range(1, num_episodes+1):
            total_loss = 0
            progress_bar = tqdm(loader, desc=f"Episode {episode}/{num_episodes}")
            
            for step, (input_ids, attention_mask) in enumerate(progress_bar):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                with torch.cuda.amp.autocast():
                    # Generate resume-aligned text
                    generated_ids = self.policy_net.transformer.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=300,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Calculate enhanced rewards
                    rewards = self.calculate_rewards(generated_ids)
                    
                    # Policy gradient optimization
                    states = generated_ids[:, :-1]
                    actions = generated_ids[:, 1:]
                    mask = (states != self.tokenizer.pad_token_id).float()
                    
                    logits = self.policy_net(states, attention_mask=mask)
                    probs = torch.softmax(logits, dim=-1)
                    selected_probs = probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                    loss = -torch.mean(torch.log(selected_probs + 1e-10) * rewards)
                    loss = loss / grad_accum

                # Gradient accumulation
                self.scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * grad_accum:.3f}",
                    'avg_loss': f"{total_loss/(step+1):.3f}"
                })

            if episode % save_every == 0:
                self.save_model(f"./resume_model_ep{episode}")
                print(f"Saved checkpoint at episode {episode}")

    # ----------------------
    # Enhanced Reward Function
    # ----------------------
    def calculate_rewards(self, generated_ids):
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        rewards = []
        resume_keywords = [
            'Kenmark', 'Smaaash', 'MELO', 'Omarun Pharma', 'Vikram Auto',
            'React.js', 'Node.js', 'MongoDB', 'Bootstrap', 'Excel Technical'
        ]
        
        for text in decoded:
            reward = 0.0
            # Structural rewards
            if any(section in text for section in ["Work Experience", "Education", "Technical Skills"]):
                reward += 1.2
            # Content rewards
            for keyword in resume_keywords:
                if keyword in text:
                    reward += 0.8
            # Formatting rewards
            if '\n-' in text or '\n•' in text:  # List formatting
                reward += 0.5
            # Penalties
            if len(text) < 100:  # Minimum length
                reward -= 1.5
            if 'undefined' in text.lower():
                reward -= 2.0
                
            rewards.append(reward)
            
        return torch.tensor(rewards, device=self.device).unsqueeze(-1)

    # ----------------------
    # Generation Methods
    # ----------------------
    def generate_resume_section(self, prompt, max_length=300, temperature=0.7):
        self.policy_net.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.policy_net.transformer.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def interactive_generation(self):
        print("Resume Generation Mode (type 'exit' to quit)")
        while True:
            prompt = input("\nEnter section to generate (e.g., 'Work Experience at Kenmark:'): ")
            if prompt.lower() == 'exit':
                break
            generated = self.generate_resume_section(prompt)
            print("\nGenerated Section:")
            print("-"*50)
            print(generated)
            print("-"*50)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.policy_net.transformer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path):
        instance = cls()
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path)
        instance.policy_net = PolicyNetwork(model_name=path).to(instance.device)
        return instance

# ----------------------
# Execution
# ----------------------
if __name__ == "__main__":
    # Initialize and train
    generator = RLResumeGenerator()
    generator.train(
        num_episodes=150,
        batch_size=4,
        save_every=50,
        grad_accum=2
    )
    
    # Save final model
    generator.save_model("./resume_generator_final")
    
    # Interactive testing
    trained_generator = RLResumeGenerator.load_model("./resume_generator_final")
    trained_generator.interactive_generation()