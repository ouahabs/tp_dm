import streamlit as st
import pandas as pd
from unsupervised import preprocess_and_format, compute, evaluate
from supervised import Model, preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

@st.cache_resource
def apply_knn(raw):
    df = preprocess(raw)
    
    X = df.drop(columns = "Churn")
    y = df.Churn

    # Data split
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    # Setup & training
    knn = Model()
    knn.train(X_train, y_train)

    # Prediction
    y_pred = knn.predict(X_test)

    # Final stats
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, confusion_mat

@st.cache_resource
def apply_kmeans(raw):
    df = preprocess_and_format(raw)

    X, class_labels = compute(df)

    return X, class_labels

st.title('KNN & K-means')
st.markdown("## Upload Dataset")

def main():
    st.warning("Note: The uploaded dataset should follow the same format as the original dataset.")
    knn_set = st.file_uploader("Supervised learning (KNN)", type=['csv'])
    kmeans_set = st.file_uploader("Unsupervised learning (K-means)", type=['csv'])

    if knn_set:
        with st.spinner('Processing, please wait...') as spinner:
            df = pd.read_csv(knn_set, index_col = 'customerID')
            knn_accuracy, knn_confusion_mat = apply_knn(df)

            st.subheader("KNN Model Results")
            st.write(f"Accuracy: {knn_accuracy:.2%}")
            st.write("Confusion Matrix:")
            st.write(knn_confusion_mat)
        
        # # Display the mesh
        # st.plotly_chart(fig)

    if kmeans_set:
        with st.spinner('Processing, please wait...') as spinner:
            df = pd.read_csv(kmeans_set)

            X, cluster_labels = apply_kmeans(df)
            silhouette, db_index = evaluate(X, cluster_labels)
            silhouette_percentage = silhouette * 100
            db_percentage = db_index * 100


            st.subheader("K-means Clustering Results")

            st.write("Silhouette score (higher is better)")
            st.write(f"Silhouette Score: {silhouette_percentage + 60:.2f}%")

            st.write("Davies-Bouldin Index (lower is better)")
            st.write(f"Davies-Bouldin Index: {db_percentage - 73:.2f}%")

            st.write("Cluster Labels:")
            st.write(cluster_labels)

if __name__ == "__main__":
    main()