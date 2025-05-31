import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from concrete import fhe
import tqdm

def define_euclidean_circuit():
    @fhe.compiler({"input1": "encrypted", "input2": "encrypted"})
    def euclidean_distance(input1, input2):
        return np.sum((input1 - input2) ** 2)
    return euclidean_distance


def evaluate_lfw(embeddings: np.ndarray,
                 labels: np.ndarray,
                 n_folds: int = 10,
                 pca_dim: int = 44,
                 quantization_strategy="global",
                 bits = 4,
                 fpr_targets: list = [0.0001, 0.001, 0.01]):

    input_sets = np.load("./data/inputsets.npz")
    # Split embeddings into pairs
    e1 = embeddings[0::2]
    e2 = embeddings[1::2]
    labels = np.asarray(labels, dtype=bool)
    n_pairs = labels.shape[0]
    fold_size = n_pairs // n_folds

    # Store TPRs for each fold and target
    tprs = np.zeros((len(fpr_targets), n_folds))

    inputset_name = f"{dim}_bit{bits}_signed"
    inputset = input_sets[inputset_name]
    inputset = [(pair[0], pair[1])for pair in inputset]

    circuit = define_euclidean_circuit().compile(inputset) #type:ignore
    mbw     = circuit.graph.maximum_integer_bit_width()
    circuit.keygen()
    print(f"MBW: {mbw}")

    # Cross-validation
    for fold in tqdm.tqdm(range(n_folds)):
        start = fold * fold_size
        end = start + fold_size
        test_idx = np.arange(start, end)
        train_idx = np.concatenate((np.arange(0, start), np.arange(end, n_pairs)))

        # Train PCA on all embeddings from training folds
        train_feats = np.vstack((e1[train_idx], e2[train_idx]))
        pca = PCA(n_components=pca_dim)
        pca.fit(train_feats)

        # Transform all embeddings
        proj1 = pca.transform(e1)
        proj2 = pca.transform(e2)

        # Normalize using bounds from training projections
        train_proj = np.vstack((proj1[train_idx], proj2[train_idx]))


        if quantization_strategy == "global":
            max_qval = 2**bits - 1
            dtype = np.int8 if bits <= 8 else np.int16
            scale = max_qval
            quantized1 = np.round(proj1  * scale).astype(dtype)
            quantized2 = np.round(proj2  * scale).astype(dtype)
        else:
            min_vals = np.min(train_proj, axis=0)
            max_vals = np.max(train_proj, axis=0)

            # Compute scale per feature
            scale = (2**bits - 1) / (max_vals - min_vals)
            
            # Avoid division by zero
            constant_features = (max_vals == min_vals)
            scale[constant_features] = 1.0

            # Quantize
            dtype = np.int8 if bits <= 8 else np.int16
            quantized1 = np.round(proj1 * scale).astype(dtype)
            quantized2 = np.round(proj2 * scale).astype(dtype)

        dists = []
        for i in range(n_pairs):
            res = circuit.encrypt_run_decrypt(quantized1[i], quantized2[i])
            dists.append(res)

        dists = np.array(dists)
        # Prepare training distances for threshold selection
        train_dists = dists[train_idx]
        train_labels = labels[train_idx]
        neg_train = train_dists[~train_labels]

        # Test distances and labels
        test_dists = dists[test_idx]
        test_labels = labels[test_idx]

        # Evaluate each FPR target
        for j, fpr in enumerate(fpr_targets):
            # Choose threshold so that fraction of neg_train < thr equals target FPR
            thr = np.percentile(neg_train, 100 * fpr)
            # True-positive rate on test set
            pos = test_labels
            tprs[j, fold] = np.mean(test_dists[pos] < thr)

    # Aggregate results
    mean_tprs = tprs.mean(axis=1)
    std_tprs = tprs.std(axis=1)


    # Display results
    for i, fpr in enumerate(fpr_targets):
        print(f"TPR @ FPR={fpr*100:.2f}%: {mean_tprs[i]:.4f} ± {std_tprs[i]:.4f}")

    return mean_tprs, std_tprs, mbw


if __name__ == "__main__":
    # Load embeddings and labels
    data = np.load("./data/pair_embeddings_ceci.npz")
    embeddings = data["embeddings"]
    labels = data["issame_list"]

    dimensions = range(40,45)
    dimensions = [44]
    qs = ["global", "per-dimension"]
    qs = qs[0]
    bits =[4,5]
    results = []
    for dim in dimensions:
        for bs in bits:
            print()
            print(f"Results with quantization: {qs}, dimension: {dim}, bits: {bs}")
            mean_tprs, std_tprs,mbw = evaluate_lfw(embeddings, labels, pca_dim=dim,
                     bits=bs,
                     quantization_strategy=qs)
            results.append({
                "dim": dim,
                "BW":bs,
                "MBW":mbw,
                "TPR@0.01%": str(round(mean_tprs[0] * 100, 2))+"$\\pm$"+str(round(std_tprs[0] * 100, 2)),
                "TPR@0.1%": str(round(mean_tprs[1] * 100, 2))+"$\\pm$"+str(round(std_tprs[1] * 100, 2)),
                "TPR@1%": str(round(mean_tprs[2] * 100, 2))+"$\\pm$"+str(round(std_tprs[2] * 100, 2)),
            })
    df = pd.DataFrame(results)
    df.to_csv(f"./ciphertext-domain/results/fold_eval_no_minmax_{qs}_rerun.csv", index=False)
    print(f"\nSaved results to fold_eval_no_minmax_{qs}_rerun.csv")
