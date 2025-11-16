import torch


def calculate_similarity(feature_matrix, mask_matrix):
    """
    Calculate similarity matrix for each batch
    @param feature_matrix: [batch_size, num_of_features, feature_dim]
    @param mask_matrix: [batch_size, num_of_features]
    """

    size = feature_matrix.size(1)  # Size of the second dimension

    matrix_expanded = feature_matrix.unsqueeze(2).expand(-1, -1, size, -1)

    matrix_transposed = feature_matrix.unsqueeze(1).expand(-1, size, -1, -1)

    # computing cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        matrix_expanded, matrix_transposed, dim=-1
    )

    mask_matrix_index_row = (mask_matrix == 0).unsqueeze(2).expand(-1, -1, 5)
    mask_matrix_index_col = (mask_matrix == 0).unsqueeze(1).expand(-1, 5, -1)

    cosine_similarity[mask_matrix_index_row] = 1
    cosine_similarity[mask_matrix_index_col] = 1

    return cosine_similarity
