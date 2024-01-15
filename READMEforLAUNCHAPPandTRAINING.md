
### Running locally
'''
cd animalguessinggame
pip install -r requirements/dev.txt
npm install
npm run-script build
npm start  # run the webpack dev server and flask server using concurrently
```

Go to `http://localhost:5000`. You will see a pretty welcome screen.

#### in case of problem with database, remove folder migrations and dev.db in folder instance

Once you have installed your DBMS, run the following to create your app's
database tables and perform the initial migration

```bash
flask db init
flask db migrate
flask db upgrade
'''

### Run classifier 
training_animals10 :

- download database on : https://www.kaggle.com/datasets/alessiocorrado99/animals10
- modify the line with your actual path for database, you should be in the file with all the folder of different animals : folder_path = "path/to/the/database/raw_img"  
- execute the code, it's training 

training_animals90 :

- download database on : https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
- modify the line with your actual path for database, you should be in the file with all the folder of different animals : folder_path = "path/to/the/database/animals"  
- execute the code, it's training 

training_MNIST :
- no need to download data, there are with the torch package
- execute the code, it's training 

VAE_cat : 
- download database on : https://www.kaggle.com/datasets/crawford/cat-dataset
- modify the line with your actual path for database, you should be in the file with all the folder of different    animals : data_path = "/chemin/vers/donnees"
modfify the path where you want to put the model and plots of training path = "/chemin/vers/fichier/ou/enregsitrement/model_figures"

- execute the code, it's training 


