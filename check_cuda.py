import torch
import tensorflow as tf

def check_cuda():
    # Check CUDA availability in PyTorch
    pytorch_cuda = torch.cuda.is_available()
    
    # Check CUDA availability in TensorFlow
    tf_cuda = tf.config.list_physical_devices('GPU')
    
    print("CUDA Availability:")
    print(f"PyTorch: {'Available' if pytorch_cuda else 'Not Available'}")
    print(f"TensorFlow: {'Available' if tf_cuda else 'Not Available'}")
    
    if pytorch_cuda:
        print(f"PyTorch is using: {torch.cuda.get_device_name(0)}")
    
    if tf_cuda:
        print("TensorFlow detected GPUs:")
        for device in tf_cuda:
            print(device)

if __name__ == "__main__":
    check_cuda()
