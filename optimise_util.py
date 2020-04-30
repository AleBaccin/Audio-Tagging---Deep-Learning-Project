from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import os

def optimise(model, traindf, params):
    datagen=ImageDataGenerator(rescale=1./255.)
    class_indices = {}

    number_of_splits = 5

    kfold_validation = KFold(n_splits = number_of_splits)

    for i, (train_split_indexes, test_split_indexes) in enumerate(kfold_validation.split(traindf)):
        train_fold = traindf.iloc[train_split_indexes]
        val_fold = traindf.iloc[test_split_indexes]

        train_generator=datagen.flow_from_dataframe(
            dataframe=train_fold,
            directory=os.path.join('images', 'train'),
            x_col="fname",
            y_col="label",
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(64,64))

        valid_generator=datagen.flow_from_dataframe(
            dataframe=val_fold,
            directory=os.path.join('images', 'train'),
            x_col="fname",
            y_col="label",
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(64,64))

        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    #Set Arbitrary placeholder variables
    val_acc_max = 0
    opt_dropout = 1
    opt_lr = 10


    #Optimise Learning Rate
    while float(params["lr"][0]) > 0.0001:
        print('Optimising LR, LR = ' + str(params["lr"][0]) + '\n')

        #Fit the training data with one epoch
        model_opt = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1)

        #Collect the validation accuracy
        val_acc = model_opt.history['val_accuracy']

        #Check if the validation accuracy is higher than the current best
        if float(val_acc[0]) > val_acc_max:

            #Update placeholders with the highest validation accuracy achieved
            #And the learning rate that achieved this
            val_acc_max = float(val_acc[0])
            opt_lr = params["lr"][0]

        print('\nvalidation accuracy for this run is: ' + str(val_acc[0]) + ' and highest accuracy achieved is: ' + str(val_acc_max) + '\n')
        #Update the learning rate to the next test sample
        params["lr"][0] *= 0.1

    #Set the learning rate to the optimised value
    params["lr"][0] = opt_lr


    #Optimise Dropouts, for loop sets which layer we are optimising for
    for i in range(0, len(params["dropouts"])):
        #Set Arbitrary placeholder variables
        val_acc_max = 0
        opt_dropout = 1

        #Fit the training data with one epoch
        while params["dropouts"][i] > 0.05:
            
            
            print('Optimising layer ' + str(i) + ' droupout, dropout = ' + str(params["dropouts"][i]) + '\n')
            model_opt = model.fit(train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=1)
            #Collect the validation accuracy
            val_acc = model_opt.history['val_accuracy']

            #Check if the validation accuracy is higher than the current best
            if float(val_acc[0]) > val_acc_max:

                #Update placeholders with the highest validation accuracy achieved
                #And the dropout that achieved this
                val_acc_max = float(val_acc[0])
                opt_dropout = params["dropouts"][i]

            print('\nvalidation accuracy for this run is: ' + str(val_acc[0]) + ' and highest accuracy achieved is: ' + str(val_acc_max) + '\n')
            #Update the dropout to the next test sample
            params["dropouts"][i] -= 0.05

        #Set the dropout to the optimised value
        params["dropouts"][i] = opt_dropout

    return params 