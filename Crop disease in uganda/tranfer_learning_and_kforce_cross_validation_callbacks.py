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

#              LEARNING RATE SCHEDULER
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