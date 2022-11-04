
# Greedy Algorithm - Gray Encoding

Algoritmo di ottimizzazione greedy per l'allenamento di una rete neurale.
I pesi della rete sono allocati in una matrice che contiene tutti i parametri di tutti i layer.

Una volta generata la configurazione iniziale i pesi vengono passati uno alla volta e si performa su ciascuno il flip di ogni di singolo bit. 

Le mosse possibili sono dunque n_pesi*n_bit, di tutto l'intorno si salvano i valori/ il valore minimo. Nel caso in cui ci siano più valori di minimo viene innescato un random walk che porta spinge a un campionamento che in alcuni casi permette di uscire dal minimo locale. 
# Loss function
![loss1](https://user-images.githubusercontent.com/83760901/199943120-19141752-3659-4900-a439-ec8d106c8bfb.png)
Orange-line: accepted configuration
Blue-line: visited configuration

![loss2](https://user-images.githubusercontent.com/83760901/199943264-e79829b5-48b9-4873-a5dc-a52335bb3760.png)
Best configuration is choosed among the visited ones.

# Accuracy on Test-Set
![example_accuracy2](https://user-images.githubusercontent.com/83760901/199943038-9096bde3-8982-4507-8e19-6e8a8f790ef9.png)
