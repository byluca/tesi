"""
A set of metrics functions.
"""
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


__all__ = ['calculate_ssim', 'calculate_psnr', 'calculate_ncc', 'calculate_node_dice', 'calculate_node_hausdorff']


def calculate_ssim(reconstructed, input):
	"""
	Structural Similarity Index (SSIM) evaluates the perceptual similarity between two images, focusing on luminance, contrast, and structural information. A higher SSIM value means the reconstructed image is more similar to the original in terms of structure and visual perception.
	SSIM values range from 0 to 1, where 0 means no similarity and 1 means perfect similarity (the images are identical).
	Function works with single images or batches.
	Args:
		reconstructed (monai.data.meta_tensor.MetaTensor | numpy.ndarray): 4D predicted image.
		input (monai.data.meta_tensor.MetaTensor | numpy.ndarray): 4D truth image.
	Returns:
		ssim (numpy.float64): the computed metric.
	"""
	input_np = input.cpu().numpy()
	reconstructed_np = reconstructed.cpu().numpy()
	if len(input_np.shape) == 4:
		return ssim(
			input_np[0, :, :, :],
			reconstructed_np[0, :, :, :],
			data_range=reconstructed_np[0, :, :, :].max() - reconstructed_np[0, :, :, :].min()
		)
	else:
		ssim_values = []
		for i in range(input_np.shape[0]):  # Loop over batch
			ssim_value = ssim(
				input_np[i, 0, :, :, :],
				reconstructed_np[i, 0, :, :, :],
				data_range=reconstructed_np[i, 0, :, :, :].max() - reconstructed_np[i, 0, :, :, :].min()
			)
			ssim_values.append(ssim_value)
		return np.mean(ssim_values)  # Return average SSIM across batch


def calculate_psnr(reconstructed, input):
	"""
	Peak Signal-to-Noise Ratio (PSNR) measures the ratio between the maximum intensity value and the reconstruction error. A higher PSNR value indicates better reconstruction quality because the reconstructed image is closer to the original.
	Function works with single images or batches.
	Args:
		reconstructed (monai.data.meta_tensor.MetaTensor | numpy.ndarray): 4D predicted image.
		input (monai.data.meta_tensor.MetaTensor | numpy.ndarray): 4D truth image.
	Returns:
		psnr (numpy.float64): the computed metric.
	"""
	mse = torch.mean((input - reconstructed) ** 2)
	max_pixel = torch.max(torch.max(reconstructed), torch.max(input))
	psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
	return psnr.item()


def calculate_ncc(reconstructed, input):
	"""
	Normalized Cross-Correlation (NCC) quantifies the similarity between the intensity patterns of two images by comparing the correlation between their voxel intensities.
	NCC values range from -1 to 1 and a higher NCC value (closer to 1) indicates that the intensity patterns in the reconstructed image closely match those of the original image.
	Function works with single images or batches.
	Args:
		reconstructed (monai.data.meta_tensor.MetaTensor | numpy.ndarray): 4D predicted image.
		input (monai.data.meta_tensor.MetaTensor | numpy.ndarray): 4D truth image.
	Returns:
		ncc (numpy.float64): the computed metric.
	"""
	if len(input.shape) == 4:
		reconstructed = torch.squeeze(torch.tensor(reconstructed))
		input = torch.squeeze(torch.tensor(input))
	input_flat = input.view(input.size(0), -1)  # Flatten per batch
	reconstructed_flat = reconstructed.view(reconstructed.size(0), -1)
	input_mean = torch.mean(input_flat, dim=1, keepdim=True)
	reconstructed_mean = torch.mean(reconstructed_flat, dim=1, keepdim=True)
	numerator = torch.sum((input_flat - input_mean) * (reconstructed_flat - reconstructed_mean), dim=1)
	denominator = torch.sqrt(torch.sum((input_flat - input_mean) ** 2, dim=1) *
							torch.sum((reconstructed_flat - reconstructed_mean) ** 2, dim=1))
	ncc = torch.mean(numerator / (denominator + 1e-10))  # Average NCC over batch
	return ncc.item()


def calculate_node_dice(y_pred, y_true, num_classes = 4, epsilon = 1e-6):
	"""
	Compute the Dice score for graph-based node predictions.
	Args:
		y_pred (torch.Tensor): predicted probabilities.
		y_true (torch.Tensor): ground truth labels.
		num_classes (int): number of classes.
		epsilon (float): small value to avoid division by zero.
	Returns:
		dice_scores (list): dice score for each class.
	"""
	dice_scores = torch.zeros(num_classes, device=y_pred.device)
	for cls in range(num_classes):
		# Create binary masks for the current class
		pred_mask = (y_pred == cls).float()
		true_mask = (y_true == cls).float()
		# Compute Dice score for the class
		intersection = torch.sum(pred_mask * true_mask)
		union = torch.sum(pred_mask) + torch.sum(true_mask)
		dice_score = (2 * intersection + epsilon) / (union + epsilon)
		dice_scores[cls] = dice_score
	return dice_scores
