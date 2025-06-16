"""
Main script to run all three tasks
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Deep Learning Course Project - THE2')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True,
                        help='Task to run (1: Deformable Conv, 2: CNN, 3: RNN)')
    parser.add_argument('--train', action='store_true',
                        help='Train the model (for tasks 2 and 3)')
    parser.add_argument('--hyperparameter-search', action='store_true',
                        help='Run hyperparameter search (task 2 only)')
    
    args = parser.parse_args()
    
    if args.task == 1:
        print("Running Task 1: Deformable Convolution Test")
        print("=" * 60)
        from task1_deformable_conv.test_deformable_conv import test_deformable_conv
        success = test_deformable_conv()
        if success:
            print("\n✅ Task 1 completed successfully!")
        else:
            print("\n❌ Task 1 failed validation")
            
    elif args.task == 2:
        print("Running Task 2: CNN for CIFAR-100")
        print("=" * 60)
        
        if args.hyperparameter_search:
            print("Running hyperparameter search...")
            from task2_cnn.hyperparameter_search import hyperparameter_search
            results, best_params, final_history = hyperparameter_search()
            
        elif args.train:
            print("Training CNN model...")
            from task2_cnn.train import train_model
            from task2_cnn.models import CustomCNN
            
            history = train_model(
                model_class=CustomCNN,
                num_epochs=30,
                learning_rate=0.01,
                optimizer_type='SGD'
            )
            
            print(f"\nFinal Test Accuracy:")
            print(f"Top-1: {history['test_top1']:.2f}%")
            print(f"Top-5: {history['test_top5']:.2f}%")
            
        else:
            print("Please specify --train or --hyperparameter-search")
            
    elif args.task == 3:
        print("Running Task 3: RNN with torch.autograd")
        print("=" * 60)
        
        from task3_rnn.train_rnn import train_manual_rnn
        manual_rnn, gradients, grade = train_manual_rnn()
        
        if grade == 30:
            print("\n✅ Task 3 completed successfully!")
        else:
            print(f"\n⚠️ Task 3 partial completion: {grade}/30 points")
    
    print("\nDone!")


if __name__ == "__main__":
    main()