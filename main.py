import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3 # Keras implementation often used for GoogLeNet concepts
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split # Still useful for initial file list splitting if not using flow_from_directory directly for splitting
from sklearn.preprocessing import LabelEncoder # To convert string labels to integers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration & Parameters ---
# GoogLeNet/InceptionV3 often uses 299x299, but 224x224 is also common
IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3) # 3 for RGB channels
BATCH_SIZE = 32  # Adjust based on GPU memory
EPOCHS = 25      # Number of training epochs (start with a moderate number, tune later)
LEARNING_RATE = 0.0001 # Learning rate for the optimizer

# --- Stage 1: Pre-processing (Simplified for CNN) ---
# Preprocessing is now tightly integrated with data loading and the model's requirements

def preprocess_image_for_cnn(image_path, target_size):
    """Loads, resizes, and prepares an image for InceptionV3."""
    try:
        print(f"Preprocessing: {image_path}")
        img = load_img(image_path, target_size=target_size) # Loads as PIL image
        img_array = img_to_array(img) # Converts to numpy array (H, W, C)

        # Ensure it's RGB (although load_img usually handles this)
        if img_array.shape[2] == 1: # Grayscale
             img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4: # RGBA
             img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Expand dimensions to add batch dimension (1, H, W, C)
        img_batch = np.expand_dims(img_array, axis=0)

        # Apply InceptionV3 specific preprocessing
        # This scales pixel values to the range [-1, 1]
        img_preprocessed = inception_v3_preprocess_input(img_batch)

        return img_preprocessed, img_array # Return preprocessed (for model) and original array (for display)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

# --- Stage 2: Segmentation (Typically NOT used for direct CNN input) ---
# Skipping K-Means/FCM for this CNN approach. The CNN learns features spatially.

# --- Stage 3: Feature Extraction (Handled by the CNN layers) ---
# Skipping manual Gabor/GLCM feature extraction.

# --- Stage 4: Classification (GoogLeNet/InceptionV3 Transfer Learning) ---

def build_transfer_model(input_shape, num_classes):
    """Builds a transfer learning model using pre-trained InceptionV3."""
    print("Building transfer learning model...")

    # Load the pre-trained InceptionV3 model without its top classification layer
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model so we don't retrain them initially
    base_model.trainable = False

    # Create the new model on top
    inputs = keras.Input(shape=input_shape)
    # We need to apply the same preprocessing as the base model expects *inside* the model
    # if we didn't do it during data loading. Here we assume it's done in loading.
    x = base_model(inputs, training=False) # Important: Use training=False for frozen layers

    # Add our custom layers for classification
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x) # Pool features
    # x = layers.Dropout(0.5)(x) # Optional Dropout for regularization
    x = layers.Dense(1024, activation='relu', name='dense_1')(x) # Example dense layer
    # x = layers.Dropout(0.5)(x) # Optional Dropout
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x) # Final classification layer

    model = keras.Model(inputs, outputs)

    print("Model Summary:")
    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', # Use categorical for one-hot labels
                  metrics=['accuracy'])

    print("Model built and compiled.")
    return model

def train_model(model, train_generator, validation_generator, epochs, class_weight=None):
    """Trains the compiled Keras model."""
    print("Starting model training...")

    # Callbacks for saving the best model and stopping early if needed
    model_checkpoint = ModelCheckpoint(
        'best_googlenet_fish_disease_model.keras', # Use .keras format
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5, # Stop after 5 epochs with no improvement in validation loss
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[model_checkpoint, early_stopping],
        class_weight=class_weight # Pass class weights if using them
    )
    print("Training complete.")
    return history

def plot_training_history(history):
    """Plots accuracy and loss curves for training and validation."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc)) # Use actual epochs completed

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()


# --- Main Workflow Example ---
if __name__ == "__main__":

    # --- 1. Dataset Loading and Preparation ---
    dataset_path = r'C:\Users\itspr\PycharmProjects\pythonProject1\NewDataset' # CHANGE THIS
    # Check if dataset path exists
    if not os.path.isdir(dataset_path):
         print(f"Error: Dataset path '{dataset_path}' not found. Please change the path.")
         print("Cannot proceed without a dataset for CNN training.")
         exit()

    # Use ImageDataGenerator for loading, augmenting (optional), and preprocessing
    # We'll apply the InceptionV3 preprocessing using its function
    datagen = ImageDataGenerator(
        preprocessing_function=inception_v3_preprocess_input,
        validation_split=0.25, # Split 25% for validation directly
        # --- Optional Augmentations (Apply only to training data generator below) ---
        # rotation_range=30,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode='nearest'
    )

    train_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3_preprocess_input,
        validation_split=0.25, # Need same split ratio
        # Add augmentations ONLY here
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    print("Setting up Training Data Generator...")
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical', # For multi-class classification with categorical_crossentropy
        subset='training',       # Specify this is the training set
        shuffle=True             # Shuffle training data
    )

    print("Setting up Validation Data Generator...")
    validation_generator = datagen.flow_from_directory( # Use the datagen *without* augmentation
        dataset_path,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',     # Specify this is the validation set
        shuffle=False            # No need to shuffle validation data
    )

    # Get class names and number of classes
    # Important: Ensure disease_types order matches the alphabetical order from flow_from_directory
    # Or rely on the generator's class_indices attribute
    class_indices = train_generator.class_indices
    # Sort class names based on indices to ensure correct order for prediction mapping
    disease_types = sorted(class_indices, key=class_indices.get)
    num_classes = len(disease_types)
    print(f"Found {train_generator.samples} training images belonging to {num_classes} classes.")
    print(f"Found {validation_generator.samples} validation images.")
    print("Class mapping:", class_indices)
    print("Disease types (ordered):", disease_types)

    if num_classes < 2:
        print("Error: Need at least two classes (directories) in the dataset path.")
        exit()

    # Optional: Calculate class weights for imbalanced datasets
    # from sklearn.utils import class_weight
    # class_weights_calculated = class_weight.compute_class_weight(
    #     'balanced',
    #     classes=np.unique(train_generator.classes), # Get unique class indices (0, 1, ...)
    #     y=train_generator.classes # Get all class indices for the training set
    # )
    # class_weight_dict = dict(enumerate(class_weights_calculated))
    # print("Calculated class weights:", class_weight_dict)
    class_weight_dict = None # Set to None if not using


    # --- 2. Build or Load the Model ---
    model_filename = 'googlenet_fish_disease_model.keras' # Use .keras extension

    # Option 1: Build and Train a new model
    # model = build_transfer_model(INPUT_SHAPE, num_classes)
    # history = train_model(model, train_generator, validation_generator, EPOCHS, class_weight=class_weight_dict)
    # plot_training_history(history) # Visualize training
    # # Save the final trained model (overwrites the best checkpoint if desired, or save separately)
    # model.save(model_filename)
    # print(f"Final model saved to {model_filename}")
    # # Load the best model saved by the checkpoint for evaluation/prediction
    # print(f"Loading best model from checkpoint: best_googlenet_fish_disease_model.keras")
    # model = tf.keras.models.load_model('best_googlenet_fish_disease_model.keras')


    # Option 2: Load a pre-trained model (if it exists)
    try:
        print(f"Attempting to load pre-trained model from {model_filename}...")
        model = tf.keras.models.load_model(model_filename)
        print("Loaded pre-trained model successfully.")
        # You might want to re-compile if loading an older format or changing optimizer settings
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    except (OSError, IOError) as e:
        print(f"No pre-trained model found at {model_filename} or error loading: {e}. Building and training a new one.")
        model = build_transfer_model(INPUT_SHAPE, num_classes)
        history = train_model(model, train_generator, validation_generator, EPOCHS, class_weight=class_weight_dict)
        plot_training_history(history) # Visualize training

        # Load the best model saved by the checkpoint instead of the potentially overfit final epoch model
        best_model_path = 'best_googlenet_fish_disease_model.keras'
        if os.path.exists(best_model_path):
             print(f"Loading best model from checkpoint: {best_model_path}")
             model = tf.keras.models.load_model(best_model_path)
             # Optionally save this best model under the main filename if desired
             # model.save(model_filename)
             # print(f"Best model saved as {model_filename}")
        else:
             print("Warning: Best model checkpoint file not found. Using model from final epoch.")
             # Save the model from the final epoch if no checkpoint exists
             model.save(model_filename)
             print(f"Model from final epoch saved to {model_filename}")


    # --- 3. Evaluate the Model ---
    print("\nEvaluating model on validation set...")
    # Use the validation generator for evaluation as test set wasn't explicitly created
    # Ideally, you'd have a separate test set generator that saw none of the training/validation data
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Generate Classification Report (more detailed metrics)
    print("\nGenerating Classification Report on validation data...")
    # Need to get predictions and true labels
    y_pred_probs = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = validation_generator.classes

    # Ensure target_names are in the correct order corresponding to class indices 0, 1, 2...
    target_names = disease_types # Use the sorted list obtained earlier

    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names, zero_division=0))


    # --- 4. Predict on a New Image ---
    new_image_path = r'C:\Users\itspr\PycharmProjects\pythonProject1\NewDataset\Redspot\23.jpg' # CHANGE THIS
    print(f"\nPredicting on a new image: {new_image_path}")

    if not os.path.exists(new_image_path):
        print(f"Error: New image path '{new_image_path}' not found. Cannot predict.")
    else:
        try:
            # Preprocess the new image using the SAME function as training
            preprocessed_img, original_img_array = preprocess_image_for_cnn(new_image_path, TARGET_SIZE)

            if preprocessed_img is not None:
                # Make prediction
                prediction_probs = model.predict(preprocessed_img) # Input needs batch dimension
                predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
                confidence = np.max(prediction_probs)

                # Map index back to class name
                predicted_disease = disease_types[predicted_class_index]

                print(f"-> Predicted Disease Index: {predicted_class_index}")
                print(f"-> Predicted Disease: {predicted_disease}")
                print(f"-> Confidence: {confidence:.4f}")

                # Display the image with prediction
                plt.imshow(original_img_array.astype('uint8')) # Display the non-normalized image
                plt.title(f"Prediction: {predicted_disease} ({confidence:.2f})")
                plt.axis('off')
                plt.show()
            else:
                 print(f"Skipping prediction due to preprocessing error on {new_image_path}.")

        except Exception as e:
            print(f"An error occurred during prediction for {new_image_path}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback