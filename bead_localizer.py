
# Import libraries
import torch


# Function to generate the profile
def calculate_profile(image, variables):
    """
    Generate a 2D Gaussian profile.
    
    Arguments:
    - image: 2D tensor representing the observed image.
    - variables: dictionary containing model variables (x, y, d, A, b, shape).
    
    Returns:
    - A 2D tensor representing the Gaussian profile.
    """

    # Extract variables
    x = variables['x']
    y = variables['y']
    d = variables['d']
    A = variables['A']
    b = variables['b']
    H, W = image.shape

    # Create a meshgrid
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # Calculate the Gaussian profile
    profile =  A * torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * d**2)) + b

    # Return the profile
    return profile

# Function to calculate the loss
def log_likelihood(image, variables):
    """
    Calculate the negative log likelihood of the image given the model variables.
    
    Arguments:
    - image: 2D tensor representing the observed image.
    - variables: dictionary containing model variables (x, y, d, A, b).
    
    Returns:
    - The negative log likelihood value.
    """
    
    # Extract variables
    sigma = variables['sigma']
    
    # Generate the Gaussian model
    profile = calculate_profile(image, variables)
    
    # Calculate the loss (negative log likelihood for Gaussian noise)
    loss = (
        1 / (2 * sigma**2) * torch.sum((image - profile)**2) +
        torch.log(2 * torch.pi * sigma**2) * image.numel() / 2
    )
    
    # Return loss
    return loss

# Localization function
def localize_bead(image, learning_rate=0.001, num_steps=500):
    """
    Localize a bead in a 2D image using Gaussian fitting.

    Arguments:
    - image: 2D tensor representing the observed image.
    - learning_rate: float, learning rate for the optimizer.
    - num_steps: int, number of optimization steps.

    Returns:
    - A dictionary with optimized variables (x, y, d, A, b, sigma).
    """

    # Initialize 
    img = image.detach().numpy()
    H, W = image.shape
    x = torch.tensor(W / 2, requires_grad=True)
    y = torch.tensor(H / 2, requires_grad=True)
    d = torch.tensor(min(H, W) / 2, requires_grad=True)
    A = torch.tensor((img.std() + img.mean()), requires_grad=True)
    b = torch.tensor(img.mean(), requires_grad=True)
    sigma = torch.tensor(img.std(), requires_grad=True)
    variables = {
        'x': x,
        'y': y,
        'd': d,
        'A': A,
        'b': b,
        'sigma': sigma,
    }

    # Clamp variables to bounds
    variables['x'].data.clamp_(0, W - 1)
    variables['y'].data.clamp_(0, H - 1)
    variables['d'].data.clamp_(min=1)
    variables['sigma'].data.clamp_(min=1e-5)

    # Initialize optimizer
    optimizer = torch.optim.Adam(variables.values(), lr=learning_rate)

    # Optimization loop
    for step in range(num_steps):

        # Zero the gradients
        optimizer.zero_grad()
        
        # Calculate the loss
        loss = log_likelihood(image, variables)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        optimizer.step()

        # Print progress
        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            print(f"-- Variables:")
            for k, v in variables.items():
                print(f"---- {k}: {v.item():.4f}")

    # Return optimized variables
    return {k: v.item() for k, v in variables.items()}
    
# Visualization function
def visualize(image, variables):
    """
    Visualize the original image and the fitted Gaussian profile.
    
    Arguments:
    - image: 2D tensor representing the observed image.
    - variables: dictionary containing optimized model variables.
    
    Returns:
    - None, but displays the image and the fitted profile.
    """
    
    # Import libraries
    import matplotlib.pyplot as plt

    # Extract variables
    x = variables['x']
    y = variables['y']

    # Get the profile
    profile = calculate_profile(image, variables)
    profile = profile.detach().numpy()

    # Convert image to numpy
    image = image.detach().numpy()
    
    # Initialize figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.ion()
    plt.show()

    # Plot the original image with lines at the bead position
    ax[0].set_title('Image')
    ax[0].imshow(image, cmap='gray')
    ax[0].plot(y, x, 'ro', markersize=5)
    ax[0].set_xlabel('X-axis')
    ax[0].set_ylabel('Y-axis')

    # Plot the fitted profile
    ax[1].set_title('Profile')
    ax[1].imshow(profile, cmap='hot', extent=(0, image.shape[1], image.shape[0], 0))
    ax[1].set_xlabel('X-axis')
    ax[1].set_ylabel('Y-axis')

    # Return figure and ax
    return fig, ax

# Example usage
if __name__ == "__main__":

    # Import libraries
    import os
    import sys
    import tifffile
    import matplotlib.pyplot as plt

    # Get path from system arguments or use default
    if len(sys.argv) > 1:
        path_to_image = sys.argv[1]
    else:
        path_to_image = os.path.join(os.path.dirname(__file__), 'data', 'TEST 1.tif')

    # Load image
    image = tifffile.imread(path_to_image)
    image = torch.from_numpy(image).float()
    
    # Preprocess the image
    image = image.sum(dim=-1)     # Sum channels
    image = - image               # Flip the image (dark bead on bright background)
    image -= image.min()          # Normalize image
    image /= image.max()          # Normalize image


    # Localize the bead
    variables = localize_bead(image, learning_rate=0.05, num_steps=500)

    # Visualize the result
    fig, ax = visualize(image, variables)
    
    # Print the result
    print("Localization Result:", variables)
