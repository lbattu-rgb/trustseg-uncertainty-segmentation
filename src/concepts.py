import cv2
import numpy as np


def _normalize_score(value, low, high):
    """Normalizes a value to the range [0, 1] based on specified low and high bounds. If the value is below the low bound, it returns 0.0; if above the high bound, it returns 1.0; otherwise, it scales linearly between the two bounds."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _resize_image(image_rgb, size):
    """Ensures that the input image is in RGB format and returns the resized image to match the specified size."""
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA) #OpenCV's recommended interpolation method for shrinking images, works by pixel-area relation


def _build_regions(mask_binary):
    """Given a binary mask, this function computes three regions of interest: the perimeter of the lesion, an outer ring surrounding the lesion, and an inner core of the lesion (256x256 binary array). It uses morphological operations (dilation and erosion) to define these regions."""
    kernel = np.ones((5, 5), np.uint8) #structuring element for morphological operations, a 5x5 square of ones
    dilated = cv2.dilate(mask_binary, kernel, iterations=1) #dilation grows the mask outward by roughly a few pixels
    eroded = cv2.erode(mask_binary, kernel, iterations=1) #erosion shrinks the mask inward by roughly a few pixels
    
    #np.clip safegaurds against negative values when subtracting 
    perimeter = np.clip(dilated - eroded, 0, 1).astype(np.uint8)#subtracting the eroded (smaller) mask from the dilated (larger) mask leaves exactly the thin band of pixels which is the boundary of lesion

    outer_ring = np.clip(cv2.dilate(mask_binary, kernel, iterations=3) - dilated, 0, 1).astype(np.uint8) #outer skin just outside the lesion 
    inner_core = cv2.erode(mask_binary, kernel, iterations=3).astype(np.uint8) #eroding by 3 iterations shrinks the mask down to just its innermost, "safest," most-clearly-lesion pixels
    return perimeter, outer_ring, inner_core


def _safe_mean(values):
    """Computes the mean of an array, returning 0.0 if the array is empty to avoid division by zero errors. This is useful for calculating average values in regions that may not contain any pixels (e.g., when a mask is empty)."""
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _join_phrases(items):
    """Joins a list of strings into a human-readable phrase. For example, ['A', 'B', 'C'] becomes 'A, B, and C'. Handles lists of length 0, 1, or 2 appropriately."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def analyze_prediction_concepts(image, mean_pred, uncertainty, threshold=0.5):
    """Analyzes a model's prediction and uncertainty map to extract interpretable concepts related to the segmentation quality. It computes various metrics such as boundary contrast, edge strength, texture confusion, and uncertainty levels in different regions of the lesion. Based on these metrics, it generates a summary, explanation, and implication regarding the trustworthiness of the model's prediction."""
    image_rgb = np.array(image.convert("RGB"))
    size = mean_pred.shape[0] #reads 256 from 256x256 prediction array dynamically
    image_rgb = _resize_image(image_rgb, size) #resize original photo to match prediction size

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 #convert to grayscale and normaize pixel values to [0, 1] for easier intensity calculations
    mask_binary = (mean_pred > threshold).astype(np.uint8) #produce a hard 0/1 lesion mask

    mask_area = float(mask_binary.mean())
    perimeter, outer_ring, inner_core = _build_regions(mask_binary) #get the three spacial zones

#convert the binary mask to boolean
    boundary_pixels = perimeter.astype(bool) 
    outer_pixels = outer_ring.astype(bool) 
    core_pixels = inner_core.astype(bool)
    lesion_pixels = mask_binary.astype(bool)

#the absolute difference in average brightness between "inside the lesion" and "just outside it."
    lesion_intensity = _safe_mean(gray[lesion_pixels])
    outer_intensity = _safe_mean(gray[outer_pixels])
    boundary_contrast = abs(lesion_intensity - outer_intensity)

#Boundary Edge Strength - edge-detection gradient operator, computing the rate of brightness change in the horizontal and vertical directions respectively
#well-defined lesion edge will have high gradient magnitude, a blurry, gradual transition (fuzzy edge) will have low gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) #overall edge strength magnitude at every pixel, regardless of direction
    boundary_edge_strength = _safe_mean(gradient_mag[boundary_pixels]) #averaged specifically over boundary_pixels, measuring how sharp/well-defined is the actual edge, right where the model drew its boundary.

#Outer Texture - measures the local variance in pixel intensity in the outer skin region, which can indicate how "busy" or "textured" the background skin is. High variance suggests a complex texture that may confuse the model.
    local_mean = cv2.GaussianBlur(gray, (9, 9), 0)
    local_sq_mean = cv2.GaussianBlur(gray ** 2, (9, 9), 0)
    local_variance = np.clip(local_sq_mean - local_mean ** 2, 0.0, None)
    outer_texture = _safe_mean(local_variance[outer_pixels])

#Uncertainty by Region - computes the average uncertainty in the entire mask, the perimeter, the outer ring, and the inner core. This helps identify where the model is most uncertain about its predictions.
    total_uncertainty = float(uncertainty.mean())
    perimeter_uncertainty = _safe_mean(uncertainty[boundary_pixels])
    outer_uncertainty = _safe_mean(uncertainty[outer_pixels])
    core_uncertainty = _safe_mean(uncertainty[core_pixels])

# Fragmented Prediction - counts the number of disconnected components in the binary mask. A higher number of fragments suggests that the model's prediction is less coherent and may indicate uncertainty or errors in segmentation.
    num_components, _ = cv2.connectedComponents(mask_binary)
    fragment_count = max(0, num_components - 1)

#dictionary of concept scores, each normalized to [0, 1] based on empirically determined thresholds. The scores are weighted combinations of the computed metrics, reflecting the relative importance of each factor in assessing segmentation quality and uncertainty.
    concept_scores = {
        "low contrast boundary": _normalize_score(0.16 - boundary_contrast, 0.0, 0.16),
        "fuzzy lesion edge": _normalize_score(0.12 - boundary_edge_strength, 0.0, 0.12),
        "background skin texture confusion": min(
            1.0,
            0.65 * _normalize_score(outer_texture, 0.002, 0.02)
            + 0.35 * _normalize_score(outer_uncertainty, 0.01, 0.08),
        ),
        "fragmented prediction": min(
            1.0,
            0.7 * _normalize_score(fragment_count, 1, 5)
            + 0.3 * _normalize_score(total_uncertainty, 0.01, 0.08),
        ),
        "high perimeter uncertainty": min(
            1.0,
            0.75 * _normalize_score(perimeter_uncertainty, 0.01, 0.08)
            + 0.25 * _normalize_score(perimeter_uncertainty - core_uncertainty, 0.0, 0.05),
        ),
    }

    ranked_concepts = sorted(concept_scores.items(), key=lambda item: item[1], reverse=True)
    active_concepts = [name for name, score in ranked_concepts if score >= 0.35][:3] # filter to only concepts that cleared a 0.35 "meaningfully present" threshold, then cap at the top 3 so UI is not cluttered

#Trust Summary - generates a human-readable summary of the model's prediction reliability based on the computed metrics and active concepts. It provides an overall assessment of the prediction's stability and areas that may require manual review.
    if mask_area < 0.01:
        trust_summary = "Prediction is very limited, so the mask should be treated cautiously."
    elif total_uncertainty < 0.01 and perimeter_uncertainty < 0.015:
        trust_summary = "Prediction looks stable, with uncertainty concentrated at a low level across the lesion."
    elif perimeter_uncertainty > core_uncertainty:
        trust_summary = "Region-aware trust is strongest in the lesion core and weaker near the outer boundary."
    else:
        trust_summary = "Prediction is moderately stable, but some regions deserve manual review."

    if active_concepts:
        explanation = f"Model uncertainty is likely driven by {_join_phrases(active_concepts)}."
    else:
        explanation = "No strong visual failure mode was triggered by the post-hoc concept rules."

    if mask_area < 0.01:
        implication = "This suggests the model may not have found a clear enough lesion region to outline confidently."
    elif perimeter_uncertainty > core_uncertainty:
        implication = "This suggests the model may struggle to accurately define lesion boundaries in this case."
    elif total_uncertainty >= 0.03:
        implication = "This suggests the full mask should be reviewed carefully before trusting the lesion extent."
    else:
        implication = "This suggests the lesion outline is relatively stable, with only limited areas needing extra review."

    return {
        "concept_scores": ranked_concepts,
        "active_concepts": active_concepts,
        "summary": trust_summary,
        "explanation": explanation,
        "implication": implication,
        "metrics": {
            "boundary_contrast": boundary_contrast,
            "boundary_edge_strength": boundary_edge_strength,
            "outer_texture": outer_texture,
            "mean_uncertainty": total_uncertainty,
            "perimeter_uncertainty": perimeter_uncertainty,
            "core_uncertainty": core_uncertainty,
            "fragment_count": fragment_count,
            "mask_area": mask_area,
        },
    }
