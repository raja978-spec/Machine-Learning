#   TRANSFER LEARNING
'''
 In this we will be using existing pre-trained models
 to train our own models.

 It is more faster than training from scratch.

 In this only the dense layer of the model will
 be trained instead of tones of CNN network.

'''

#   K_FOLD CROSS VALIDATION
'''
 Here all the data will be used for training set and validation set.
 in normal random_split some percentage of data will be used for
 training set and some for validation set.

 But here we will use all the data for training set and validation set.

 It we splits data into k parts. From that some part of data used for training
 and some for validation. It actually iterates all the data to .

 REFER K_FOLD_CROSS_VALIDATION.doc for screen shot

 PyTorch doesn't have a dedicated tool for this. Instead, we'll use the KFold 
 splitting tool from scikit-learn.

 k = 5

 Here k is the chunk size, here it will uses 20%

 kfold_splitter = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

 train_nums, val_nums = next(kfold_splitter.split(range(100)))
 fold_fraction = len(val_nums) / (len(train_nums) + len(val_nums))
 print(f"One fold is {100*fold_fraction:.2f}%")

 OUTPUT:
 One fold is 20.00%

'''


#   TRAINING K-FOLD CROSS VALIDATION
'''
 from training import predict, train

 loss_fn = torch.nn.CrossEntropyLoss()
 optimizer = torch.optim.Adam(model.parameters())

 We need to reset the model parameters value before
 going to feed our input on newly replaced layer.

 Because the pre-trained model is already trained on some data
 and it has some weights and different output feature, in our
 image process we have only 5 classes to predict, it pre-trained
 model is trained on 1000 classes.

 We need to reset the model parameters value before
 going to feed our input on newly replaced layer.

 If you don't reset the classifier layers:

 1. The model will start training with its current weights.
 2. If these weights come from pre-training on a different task, it might take 
    longer to converge.
 3. The learning process might be biased toward the pre-trained task, leading to suboptimal performance.

 SUMMARY:

 Resetting the classifier layers ensures that they start with fresh weights, 
 allowing the model to learn effectively for the new task. 
 This is especially critical in transfer learning or when modifying 
 the classifier structure.

 In our model fc is the layer we replaced

 def reset_classifier(model):
    model.fc.get_submodule("0").reset_parameters()
    model.fc.get_submodule("3").reset_parameters()


 WHAT WILL HAPPEN IF YOU RESET WHOLE MODEL PARAMTERS

 1. You lose all pre-trained knowledge

 Reset only the newly added layers when performing transfer learning. 
 This allows the model to adapt to your task while leveraging the general 
 feature extraction learned during pre-training.

 num_epochs = 6

 We're ready. For k-fold, we'll train in a loop that will run 
 times. In each run, we'll have one fold as our validation set and the rest as training.

 On each loop we'll do a few things:

 1. Get which observations are in the training set and which are in the validation set from our k-fold splitter.
 2. Create a training and a validation data loader
 3. Reset the classifier part of our model
 4. Train the model with this training set and validation set
 5. Record the losses and accuracies from the training process

 # you can safely skip this cell and load the model in the next cell

 training_records = {}
 fold_count = 0

 for train_idx, val_idx in kfold_splitter.split(np.arange(len(dataset))):
    fold_count += 1
    print("*****Fold {}*****".format(fold_count))

    # Make train and validation data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Reset the model
    reset_classifier(model)

    # Train
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs=num_epochs,
        device=device,
        use_train_accuracy=False,
    )

    # Save training results for graphing
    training_records[fold_count] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    print("\n\n")

 print(type(training_records))
 training_records.keys()

 OUTPUT:

 <class 'dict'>
 dict_keys([1, 2, 3, 4, 5])

 print(type(training_records[1]))
 training_records[1].keys()

 OUTPUT:

 <class 'dict'>
 dict_keys(['train_losses', 'val_losses', 'val_accuracies'])

 training_records[1]["train_losses"]

 [1.2573940041655391,
 1.0369858069006581,
 0.9532191420728567,
 0.9031010592602149,
 0.8534171506126539,
 0.7998687047670331]

 We'll make a function to plot one kind of value for all folds.

 def plot_all_folds(data, measurement):
    for fold in data.keys():
        plt.plot(data[fold][measurement], label=f"Fold {fold}, {measurement}")
    plt.xlabel("Epochs")
    plt.legend()

 Let's try it out on the training loss.

 plot_all_folds(training_records, "train_losses")
'''

#                    CALLBACKS
'''
 Helps to modify the pre-trained training process to perform model better, it is a
 function that changes the training process.

 Three most common callbacks:

 1. Early Stopping - Stop training the model when decreased loss function 
                     started increasing.

                     So it halts training when condition met

                     early_stopping(watch='train_loss', patience=5)

                     Above function will stop training when train_loss
                     is not increased for 5 epochs.

 2. Model Checkpointing  - Saves the model every time validation loss gets 
                           better than in the epoch prior. 

                           This allows us to recover the best model 
                           once training completes.
                           
                           checkpointing(watch='train_loss')

 3. Learning Rate Scheduler - Change learning rate during training. It
                              is found in adam optimizer. Helps to
                              avoid overfitting.

                              lr_scheduler(freq, decrease_rate)
'''

#              LEARNING RATE SCHEDULER FUNCTION
'''
 For the Learning Rate Scheduling, we'll use StepLR 
 from torch.optim. The StepLR scheduler decays the 
 learning rate by multiplicative factor gamma every 
 step_size epochs.

 The multiplicative factor (gamma) in StepLR is the 
 value by which the learning rate is multiplied after 
 every step_size epochs.

 For example:

 Initial learning rate: 0.1
 gamma: 0.5
 step_size: 2
 
 After every 2 epochs:
 Epoch 1-2: Learning rate = 0.1
 Epoch 3-4: Learning rate = 0.1 × 0.5 = 0.05
 Epoch 5-6: Learning rate = 0.05 × 0.5 = 0.025
 
 So, gamma controls how much the learning rate decreases.

 from torch.optim.lr_scheduler import StepLR
 
 # Period of learning rate decay
 step_size = 4
 # Multiplicative factor of learning rate decay
 gamma = 0.2

 # Initialize the learning rate scheduler
 scheduler = StepLR(
    optimizer,
    step_size=step_size,
    gamma=gamma,
 ) 

 print(type(scheduler))
'''

#               EARLY STOPPING FUNCTION
'''
 For Early Stopping, we'll create a function early_stopping 
 that we'll call from within the train function. 
 
 The early_stopping function accepts:

 the current validation loss,
 the best validation loss so far
 the number of epochs since validation loss last improved (counter).
 In the function we need to check if validation loss improved. 
 If yes, we reset the counter. If not, we add one to the counter. 
 We also need to check if validation loss hasn't improved in the 
 last 5 epochs. If that is the case, we should set stopping to True.

 def early_stopping(validation_loss, best_val_loss, counter):
    """Function that implements Early Stopping"""

    stop = False

    if validation_loss < best_val_loss:
        counter = 0
    else:
        counter += 1

    # Check if counter is >= patience (5 epochs in our case)
    # Set stop variable accordingly
    if counter >= 5:
        stop = True

    return counter, stop
'''

#              MODEL CHECKPOINTING FUNCTION
'''
 def checkpointing(validation_loss, best_val_loss, model, optimizer, save_path):

    if validation_loss < best_val_loss:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            },
            save_path,
        )
        print(f"Checkpoint saved with validation loss {validation_loss:.4f}")
'''

#   USING CALLBACKS IN MODEL TRAINING
'''
 Now we're ready to modify the train function to include an 
 option to use Callbacks.

 Notice that the modified train function below is 
 quite similar to what we've used before. We just added scheduler, 
 checkpoint_path and early_stopping as optional arguments. 
 
 As you can see at the end of the modified train function, 
 we use these three callbacks when function is called with appropriate inputs.

 from training import score, train_epoch


 def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=20,
    device="cpu",
    scheduler=None,
    checkpoint_path=None,
    early_stopping=None,
):
    # Track the model progress over epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    # Create the trackers if needed for checkpointing and early stopping
    best_val_loss = float("inf")
    early_stopping_counter = 0

    print("Model evaluation before start of training...")
    # Test on training set
    train_loss, train_accuracy = score(model, train_loader, loss_fn, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # Test on validation set
    validation_loss, validation_accuracy = score(model, val_loader, loss_fn, device)
    val_losses.append(validation_loss)
    val_accuracies.append(validation_accuracy)

    for epoch in range(1, epochs + 1):
        print("\n")
        print(f"Starting epoch {epoch}/{epochs}")

        # Train one epoch
        train_epoch(model, optimizer, loss_fn, train_loader, device)

        # Evaluate training results
        train_loss, train_accuracy = score(model, train_loader, loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Test on validation set
        validation_loss, validation_accuracy = score(model, val_loader, loss_fn, device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy*100:.4f}%")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy*100:.4f}%")

        # # Log the learning rate and have the scheduler adjust it
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)
        if scheduler:
            scheduler.step()

        # Checkpointing saves the model if current model is better than best so far
        if checkpoint_path:
            checkpointing(
                validation_loss, best_val_loss, model, optimizer, checkpoint_path
            )

        # Early Stopping
        if early_stopping:
            early_stopping_counter, stop = early_stopping(
                validation_loss, best_val_loss, early_stopping_counter
            )
            if stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss

    return (
        learning_rates,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        epoch,
    )

    epochs_to_train = 50

train_results = train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=epochs_to_train,
    device=device,
    scheduler=scheduler,
    checkpoint_path="model/LR_model.pth",
    early_stopping=early_stopping,
)

(
    learning_rates,
    train_losses,
    valid_losses,
    train_accuracies,
    valid_accuracies,
    epochs,
) = train_results
'''

# PLOTTING ALL THE RESULTS AND CONFUSION MATRIX
'''
 The above function will be stopped early so it trains only 15 epochs.

 In below we will plot how train and val loss and accuracy changes over epochs.

 # Plot train losses, use label="Training Loss"

 plt.plot(train_losses, label="Training Loss")
 plt.plot(valid_losses, label="Validation Loss")
 plt.ylim([0, 1.7])
 plt.title("Loss over epochs")
 plt.xlabel("Epochs")
 plt.ylabel("Loss")
 plt.legend();

 In above chart output the loss function are decreased
 as expected.

 # Plot train accuracies, use label="Training Accuracy"
 plt.plot(train_accuracies, label='Training Accuray')
 # Plot validation accuracies, use label="Validation Accuracy"
 plt.plot(valid_accuracies, label="Validation Accuray")
 plt.ylim([0, 1])
 plt.title("Accuracy over epochs")
 plt.xlabel("Epochs")
 plt.ylabel("Accuracy")
 plt.legend();

 In the above chart the accuracy is also increased as expected.

 # Plot the learning rates
 plt.figure(figsize=(10, 6))
 plt.plot(range(1, epochs + 1), learning_rates, marker="o", label="Learning Rate")
 plt.title("Learning Rate Schedule")
 plt.xlabel("Epoch")
 plt.ylabel("Learning Rate")
 plt.show()

 In the above chart the learning rate is decreased as expected.

 Now we will see the saved models by checkpointing.

 checkpoint = torch.load("model/LR_model.pth")

 # Load the state dictionaries
 model.load_state_dict(checkpoint["model_state_dict"])
 optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

 from training import predict

 probabilities_val = predict(model, val_loader, device)
 predictions_val = torch.argmax(probabilities_val, dim=1)

 print(predictions_val)

 targets_val = torch.cat([labels for _, labels in tqdm(val_loader, desc="Get Labels")])

 cm = confusion_matrix(targets_val.cpu(), predictions_val.cpu())

 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

 # Set figure size
 plt.figure(figsize=(10, 8))

 disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical");

 REFER K_FOLD_CROSS_VALIDATION.doc for screen shot
'''