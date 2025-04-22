class Path_Hyperparameter:
    random_seed = 42  # Random seed

    # Training hyperparameters
    epochs: int = 200  # Number of training epochs
    batch_size: int = 16  # Batch size
    inference_ratio = 1  # Batch size for validation and testing is a multiple of the training batch size
    learning_rate: float = 1e-3  # Learning rate of 1e-3
    factor = 0.1  # Learning rate decay factor
    patience = 12  # Patience value for learning rate scheduler
    warm_up_step = 1000  # Warm-up steps. Gradually increasing the learning rate during early training to improve stability and convergence. Initially set to 1000
    weight_decay: float = 1e-3  # Weight decay for AdamW optimizer
    amp: bool = True  # Whether to use mixed precision for faster training and reduced GPU memory usage
    load: str = None  # Load model and/or optimizer from a .pth file to restore previous state during testing or continued training
    max_norm: float = 20  # Maximum gradient norm for gradient clipping. Limiting the maximum norm of gradients to prevent gradient explosion

    # Evaluation and testing hyperparameters
    evaluate_epoch: int = 0  # Start evaluating after how many epochs. No evaluation is done until the 10th epoch
    evaluate_inteval: int = 1  # Perform evaluation every specified number of epochs
    test_epoch: int = 101  # Start testing after how many epochs
    stage_epoch = [0, 0, 0, 0, 0]  # Adjust learning rate after each stage
    save_checkpoint: bool = False  # Whether to save model checkpoints
    save_interval: int = 50  # Save checkpoint every specified number of epochs
    save_best_model: bool = True  # Whether to save the best model

    # Model hyperparameters
    # RSM_SS tiny
    drop_path_rate = 0  # Drop path rate. Randomly dropping some paths in the network forces the model to use different path combinations during each forward pass, helping the model learn more robust features
    dims = 64  # Dimension? 96
    depths = [9, 2, 2, 2]  # Depth of each stage, originally 2292, but increasing depth showed minimal effect and made training extremely slow
    ssm_d_state = 16  # SSM state dimension
    ssm_dt_rank = "auto"  # SSM Dt rank. The rank of the Dt matrix determines the amount of independent information available during state transition. Used to specify or automatically determine the rank of the Dt matrix in the state space model
    ssm_ratio = 2.0  # SSM ratio. Controls the proportional relationships between different parts of the model to enhance its representational power while controlling computational complexity
    mlp_ratio = 4.0  # MLP ratio. The MLP (Multi-Layer Perceptron) consists of multiple fully connected (linear) layers. The mlp_ratio parameter controls the width (number of neurons) of hidden layers relative to the input layer width
    downsample_version = "v3"
    patchembed_version = "v2"  # Refers to dividing the input image into several small patches and then mapping these patches to a high-dimensional space

    # Data parameters
    image_size = 256  # Image size
    downsample_raito = 1  # Reduces the resolution of the image by downsampling to decrease computational complexity and memory usage. A downsample_raito of 1 means no downsampling
    dataset_name = 'SAR_Dataset_CD'  # Dataset name
    root_dir = '.'  # Root directory of the dataset (current path)

    # Inference parameters
    log_path = './log_feature/'  # Log path

    # log wandb hyperparameters
    # Run this code in the terminal to view relevant training metrics, images, etc., in the browser: wandb sync <wandb_directory>
    # log_wandb_project: str = 'train_whu_cd'  # wandb project name
    log_wandb_project: str = 'MyTraining'  # wandb project name (Weights and Biases, tracks model hyperparameters, training metrics such as loss, accuracy, model weights, and output logs)

    project_name = f'{log_wandb_project}_{image_size}_{learning_rate}'  # Project name

    # Extract all properties and their values from the Path_Hyperparameter class and return them as a dictionary. This method is commonly used for saving and loading model states
    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}
        # Return a dictionary containing all attributes and their values that do not start with '_'

ph = Path_Hyperparameter()
# Create an instance of the Path_Hyperparameter class
