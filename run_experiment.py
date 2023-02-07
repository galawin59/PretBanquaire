def run_experiment(X_train,y_train,X_test,y_test,model,modele_type,confusion=False,mlflow_tracking=True):
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    try:
        print('Best Hyperparameters: %s' % model.best_params_)
    except:
        pass

    if modele_type == "regression":
        from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
        import mlflow
        print("######## R^2 : ")
        print("TRAIN :",r2_score(y_train, y_pred_train))
        print("TEST :",r2_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_r2", r2_score(y_train, y_pred_train))
            mlflow.log_metric("test_r2", r2_score(y_test, y_pred_test))
        print("######## MAE : ")
        print("TRAIN :",mean_absolute_error(y_train, y_pred_train))
        print("TEST :",mean_absolute_error(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_mae", mean_absolute_error(y_train, y_pred_train))
            mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_pred_test))
        print("######## MSE : ")
        print("TRAIN :",mean_squared_error(y_train, y_pred_train))
        print("TEST :",mean_squared_error(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_mse", mean_squared_error(y_train, y_pred_train))
            mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred_test))
        
        

    elif modele_type == "classification":
        from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        import mlflow
        print("######## accuracy_score : ")
        print("TRAIN :",accuracy_score(y_train, y_pred_train))
        print("TEST :",accuracy_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
        print("######## f1_score : ")
        print("TRAIN :",f1_score(y_train, y_pred_train))
        print("TEST :",f1_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_f1", f1_score(y_train, y_pred_train))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))
        print("######## precision_score : ")
        print("TRAIN :",precision_score(y_train, y_pred_train))
        print("TEST :",precision_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_precision", precision_score(y_train, y_pred_train))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
        print("######## recall_score : ")    
        print("TRAIN :",recall_score(y_train, y_pred_train))
        print("TEST :",recall_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_recall", recall_score(y_train, y_pred_train))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
        print("######## roc_auc_score : ")    
        print("TRAIN :",roc_auc_score(y_train, y_pred_train))
        print("TEST :",roc_auc_score(y_test, y_pred_test))  
        if mlflow_tracking:
            mlflow.log_metric("train_roc", roc_auc_score(y_train, y_pred_train))
            mlflow.log_metric("test_roc", roc_auc_score(y_test, y_pred_test))  

        if confusion:
            
            if mlflow_tracking:
                mlflow.log_artifact("test_confusion", confusion_matrix(y_test, y_pred_test)) 
            return model.best_estimator_, confusion_matrix(y_test, y_pred_test)

    return model.best_estimator_