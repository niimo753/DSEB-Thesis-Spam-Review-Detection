import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

def cumulative_variance_ratio_plot(data):
    pca = PCA()
    pca.fit(data)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    sns.lineplot(cumulative_variance_ratio)
    plt.title("Elbow Chart for PCA")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.xlabel("Number of Principal Components")

def pca_scree_plot(data):
    pca = PCA(n_components=10)
    pca.fit(data)
    variance_ratio = pca.explained_variance_ratio_ * 100

    sns.barplot(variance_ratio, alpha=1)
    sns.lineplot(variance_ratio, alpha=1, color="grey")
    sns.scatterplot(variance_ratio)
    plt.title("Scree Plot for PCA")
    plt.ylabel("Percentage Of Explained Variances (%)")
    plt.xlabel("Principal Components")
    plt.grid(True)

def knn_visualization(x_train, y_train, x_test, y_test, model=None, y_pred=None, k_neighbors=1, show_distance=False,
                      wrong_case_only=True, indices=None, reconstruct_image=False):
    # predicted y
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    x_train_reduced, x_test_reduced = model.get_projected_data()
    print("Accuracy:", model.score(x_test, y_test))

    if wrong_case_only:
        if indices is None:
            indices = np.where(y_test != y_pred)[0]
        else:
            mis_indices = np.where(y_test != y_pred)[0]
            indices = [i for i in mis_indices if i in indices]
    else:
        if indices is None:
            indices = np.arange(len(y_test))

    predicted = model.clf.kneighbors(x_test_reduced, n_neighbors=k_neighbors, return_distance=show_distance)
    if show_distance:
        distance, predicted_indices = predicted[0], predicted[1]
    else:
        predicted_indices = predicted
    numcases = len(indices)

    if reconstruct_image:
        x_test = model.preprocessor.inverse_transform(x_test_reduced)
        x_train = model.preprocessor.inverse_transform(x_train_reduced)
    
    if numcases == 0:
        print("No misclassification found")
    
    else:
        _ , ax = plt.subplots(nrows=numcases, ncols=k_neighbors+1, figsize=((k_neighbors+1)*5, numcases*6.5))
        index = 0

        if numcases == 1:
            for i in range(len(ax)):
                if i == 0:
                    img = x_test[indices[i]]
                    label = f"True: {y_test[indices[index]]}"
                else:
                    img = x_train[predicted_indices[indices[index]][i-1]]
                    label = f"Top {i} Predicted: {y_train[predicted_indices[indices[index]][i-1]]}"
                    if show_distance:
                        ax[i].set_xlabel(f"Distance: {round(distance[indices[index]][i-1], 2)}", fontsize="xx-large")
                img = img.reshape((112, 92))
                ax[i].imshow(img, cmap='gray')
                ax[i].set_title(label, fontsize="xx-large")
                ax[i].set_xticks(())
                ax[i].set_yticks(())

        else:
            for row in ax:
                for col in range(len(row)):
                    if col == 0:
                        img = x_test[indices[index]]
                        label = f"True: {y_test[indices[index]]}"
                    else:
                        img = x_train[predicted_indices[indices[index]][col-1]]
                        label = f"Top {col} Predicted: {y_train[predicted_indices[indices[index]][col-1]]}"
                        if show_distance:
                            row[col].set_xlabel(f"Distance: {round(distance[indices[index]][col-1], 2)}", fontsize="xx-large")
                    img = img.reshape((112, 92))
                    row[col].imshow(img, cmap='gray')
                    row[col].set_title(label, fontsize="xx-large")
                    row[col].set_xticks(())
                    row[col].set_yticks(())
                index += 1