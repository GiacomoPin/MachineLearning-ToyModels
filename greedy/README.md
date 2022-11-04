
# Greedy Algorithm - Gray Encoding

Algoritmo di ottimizzazione greedy per l'allenamento di una rete neurale.
I pesi della rete sono allocati in una matrice che contiene tutti i parametri di tutti i layer.

Una volta generata la configurazione iniziale i pesi vengono passati uno alla volta e si performa su ciascuno il flip di ogni di singolo bit. 

Le mosse possibili sono dunque n_pesi*n_bit, di tutto l'intorno si salvano i valori/ il valore minimo. Nel caso in cui ci siano più valori di minimo viene innescato un random walk che porta spinge a un campionamento che in alcuni casi permette di uscire dal minimo locale. Nel caso in cui ci sia un solo valore minimo 
