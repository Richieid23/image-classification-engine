from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA

# ================== Feature Extraction ==================
def extract_pca(X_train, n_components=100):
    pca = PCA(n_components=n_components, random_state=42)
    X_flat = X_train.reshape(X_train.shape[0], -1)
    X_pca = pca.fit_transform(X_flat)
    return X_pca, pca

def extract_lda(X_train, y_train, n_components=3):
    lda = LDA(n_components=n_components)
    X_flat = X_train.reshape(X_train.shape[0], -1)
    X_lda = lda.fit_transform(X_flat, y_train)
    return X_lda, lda

def extract_ica(X_train, n_components=100):
    ica = FastICA(n_components=n_components, whiten='arbitrary-variance', max_iter=1000, tol=1e-4)
    X_flat = X_train.reshape(X_train.shape[0], -1)
    X_ica = ica.fit_transform(X_flat)
    return X_ica, ica

def extract_pca_lda(X_train, y_train, n_pca=100, n_lda=3):
    X_flat = X_train.reshape(X_train.shape[0], -1)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_flat)

    n_classes = len(np.unique(y_train))
    n_lda_final = min(n_lda, n_classes - 1)
    lda = LDA(n_components=n_lda_final)
    X_plda = lda.fit_transform(X_pca, y_train)
    return X_plda, (pca, lda)

def transform_pca_lda(pca, lda, X):
    X_flat = X.reshape(X.shape[0], -1)
    X_pca = pca.transform(X_flat)
    X_lda = lda.transform(X_pca)
    return X_lda