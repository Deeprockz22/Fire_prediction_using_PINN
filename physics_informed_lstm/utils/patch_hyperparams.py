import sys

with open('fire_predict.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''    print("Training Configuration:")
    print("  • Architecture: Physics-Informed LSTM")
    print("  • Input features: 9 (3 original + 6 physics correlations)")
    print("    - Original: HRR, Q_RADI, MLR")
    print("    - Heskestad: Flame height, growth rate, deviation")
    print("    - McCaffrey: Plume region classification")
    print("    - Thomas: Ventilation flow factor")
    print("    - Buoyancy power scaling")
    print("  • Epochs: 50 (with early stopping)")
    print("  • Batch size: 32")
    print("  • Sequence length: 30 steps")
    print("  • Prediction horizon: 10 steps\\n")
    
    response = input("Continue with training? [y/N]: ").strip().lower()'''

new_block = '''    # Default hyperparameters
    config = {
        'epochs': 50,
        'batch_size': 32,
        'hidden_dim': 128,
        'num_layers': 2,
        'learning_rate': 0.001
    }
    
    print("Training Configuration:")
    print("  • Architecture: Physics-Informed LSTM")
    print("  • Input features: 9 (3 original + 6 physics correlations)")
    print("    - Original: HRR, Q_RADI, MLR")
    print("    - Heskestad: Flame height, growth rate, deviation")
    print("    - McCaffrey: Plume region classification")
    print("    - Thomas: Ventilation flow factor")
    print("    - Buoyancy power scaling")
    print(f"  • Epochs: {config['epochs']} (with early stopping)")
    print(f"  • Batch size: {config['batch_size']}")
    print("  • Sequence length: 30 steps")
    print("  • Prediction horizon: 10 steps\\n")
    
    print("🤖 Would you like to mathematically optimize these hyper-parameters?")
    tweak = input("Tweak model settings? [y/N]: ").strip().lower()
    
    if tweak == 'y':
        print("\\n🔧 HYPERPARAMETER TERMINAL")
        print("Press Enter to keep the default value.\\n")
        
        try:
            ep = input(f"Epochs [{config['epochs']}]: ").strip()
            if ep: config['epochs'] = int(ep)
            
            bs = input(f"Batch Size [{config['batch_size']}]: ").strip()
            if bs: config['batch_size'] = int(bs)
            
            hd = input(f"Hidden Dimensions [{config['hidden_dim']}]: ").strip()
            if hd: config['hidden_dim'] = int(hd)
            
            nl = input(f"LSTM Layers [{config['num_layers']}]: ").strip()
            if nl: config['num_layers'] = int(nl)
            
            lr = input(f"Learning Rate [{config['learning_rate']}]: ").strip()
            if lr: config['learning_rate'] = float(lr)
            
            print("\\n✅ Matrix parameters re-aligned successfully.\\n")
        except ValueError:
            print("\\n❌ Invalid input detected. Reverting to safe default parameters.\\n")
    
    response = input("Continue with training using these settings? [y/N]: ").strip().lower()'''

old_vars = '''        # Configuration
        INPUT_SEQ_LEN = 30
        PRED_HORIZON = 10
        BATCH_SIZE = 32
        MAX_EPOCHS = 50'''

new_vars = '''        # Configuration
        INPUT_SEQ_LEN = 30
        PRED_HORIZON = 10
        BATCH_SIZE = config['batch_size']
        MAX_EPOCHS = config['epochs']'''

old_model = '''        # Create model
        print("🧠 Initializing model with enhanced physics correlations...")
        model = PhysicsInformedLSTM(
            input_dim=9,  # 3 original + 6 physics features
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.1,
            lr=0.001,
            pred_horizon=PRED_HORIZON,
            use_physics_loss=True,'''

new_model = '''        # Create model
        print("🧠 Initializing model with enhanced physics correlations...")
        model = PhysicsInformedLSTM(
            input_dim=9,  # 3 original + 6 physics features
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=1,
            dropout=0.1,
            lr=config['learning_rate'],
            pred_horizon=PRED_HORIZON,
            use_physics_loss=True,'''

if old_block in content and old_vars in content and old_model in content:
    content = content.replace(old_block, new_block)
    content = content.replace(old_vars, new_vars)
    content = content.replace(old_model, new_model)
    with open('fire_predict.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Hyperparameter tweaking added.")
else:
    print("ERROR: Blocks not found.")
    print("Block 1 found:", old_block in content)
    print("Block 2 found:", old_vars in content)
    print("Block 3 found:", old_model in content)
