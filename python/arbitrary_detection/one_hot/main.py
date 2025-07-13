from one_hot.encoder import OneHotEncoder

if __name__ == "__main__":
    encoder = OneHotEncoder()
    encoder.load_dataset()

    print("dataset:", len(encoder.get_positive_data()), len(encoder.get_negative_data()))
    encoder.encode_data()

    print("encoded:", len(encoder.get_positive_encoded()), len(encoder.get_negative_encoded()))

    # encoder.visualize_features()

    # encoder.train_classifier()
