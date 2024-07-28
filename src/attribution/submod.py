import numpy as np
from submodlib import FacilityLocationFunction

def subset_selection(embeddings, k):
    """
    Perform subset selection on the given passages or passage embeddings.

    Args:
    - passages (list or tuple): List of passages.
    - embeddings (np.ndarray): Embedding vectors for the passages, shape: (n, d) 
        n: No of passages
        d: Embedding dimension
    - k (int): Target subset size.

    Returns:
    - list: Selected subset of passages.
    """
   
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings should be a numpy ndarray.")
    
    if k <= 0:
        raise ValueError("k must be a positive integer and less than or equal to the number of passages.")
    
    if  k > embeddings.shape[0]:
        return [i for i in range(embeddings.shape[0])]
    
    # Compute the similarity matrix
    similarities = np.matmul(embeddings, embeddings.T)

    fl_func = FacilityLocationFunction(n=len(passages), mode="dense", sijs=similarities, separate_rep=False)
    selected_metadata = fl_func.maximize(budget=k)
    selected_indices = [i[0] for i in selected_metadata]

    return selected_indices

# Example usage:
if __name__ == "__main__":
    passages = ["passage1", "passage2", "passage3", "passage4"]
    embeddings = np.random.rand(4, 128)  # Example embeddings
    k = 2
    selected_passages = subset_selection(passages, embeddings, k)
    print(selected_passages)
