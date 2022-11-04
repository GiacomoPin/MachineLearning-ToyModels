import os
import cv2
import time
import numpy as np
from utils import sigmoid, tanh, RELU, Tanh_tab, Sigmoid_tab, activation_discrete, Loss_MSE, Loss_xentropy, Net_SA2
from quant_utils import  dec_to_4bit, bit_flip, dec_to_gray, gray_to_dec


def MLP_GREEDY(N_iter, N_input, n_neurons, N_output, N_bit_input, N_bit_weigh, N_bit_hidden, N_bit_output, dataset, X_test, t_outs_test, letter):

    #### Weights initialization--------------------------------------------------#
    #np.random.seed(203)
    if True:
        WEIGHTS = np.round(np.random.random((N_input + 1 + 1 + 1, n_neurons))*\
                                                (np.power(2, N_bit_weigh)-1))
        WEIGHTS = WEIGHTS.astype(int)
        np.savetxt(path_save + 'WEIGHTS_I', WEIGHTS, delimiter=',')
    if False:
        WEIGHTS = np.loadtxt(path_save+ 'WEIGHTS_F', delimiter=',')   
        WEIGHTS = WEIGHTS.astype(int)

    Loss      = []
    LOSS      = []
    Accuracy_test = []


    x = dataset[:, 0:N_input]
    t_outs = dataset[:, N_input: N_output + N_input] * (np.power(2,\
                         N_bit_output) - 1) - np.power(2, N_bit_output - 1)

    tabulated_sigmoid1, minimum_value1 = Tanh_tab(N_bit_input, N_bit_weigh,
                                                  N_input, N_bit_hidden)
    tabulated_sigmoid2, minimum_value2 = Tanh_tab(N_bit_hidden, N_bit_weigh,
                                                  n_neurons, N_bit_output)
    if 0:
        plt.plot(tabulated_sigmoid1)
        plt.grid(True)
        plt.show()


    random_walk=0
    #------------------------------------------------------------------------------------ 
    for i in range(N_iter):  

        # ----   ----   ----   ----   ----  ----  ----  ----  ----  ----  ----  ----  ---
        #1) Feedforward
        if i==0:    
            z0,z,y0,y = Net_SA2(x,(WEIGHTS-8),
                                tabulated_sigmoid1,minimum_value1,
                                tabulated_sigmoid2,minimum_value2)
            loss_old  = Loss_MSE(t_outs,y)
            LOSS.append(loss_old)
            print(loss_old)


        # ----   ----   ----   ----   ----  ----  ----  ----  ----  ----  ----  ----  ---
        #2) GREEDY Algorithm
        if False:  print('Iteration:',i) 

        RANDOM_WALK_row = []
        RANDOM_WALK_col = []
        NEW_WEIGHT      = []
        for row in range(len(WEIGHTS[:,0])):
            for column in range(len(WEIGHTS[0,:])):
            # Converting int to a 4bit number
            # Make a copy of weights on which perform a trial move.
                WEIGHTS_TRIAL = copy.deepcopy(WEIGHTS)
                WEIGHTS_TRIAL_bit = dec_to_4bit(\
                                    dec_to_gray(WEIGHTS_TRIAL[row,column]))
            # Converting the n-th weight into a 4bit string in gray code .  
            # Converting 4bit to int([0,15]), must be rescaled between [-8,7]
                
                if False: 
                    ''' TEST SULLA CODIFICA GRAY: 
                    Print the weight in and grey(decimal map)''' 
                    print('FLipping:')
                    print(WEIGHTS_TRIAL_bit, gray_to_dec(int(WEIGHTS_TRIAL_bit\
                                                                         ,2)))  
                    print('--------') 

                binary_list = list(WEIGHTS_TRIAL_bit)
                for bit in range(4):
                    flipped  = "".join(bit_flip(binary_list,bit))
                    w1elenew = gray_to_dec(int(flipped,2))
                    WEIGHTS_TRIAL[row,column]=w1elenew
 
#------------------------------------------------------------------------------
          ########## HERE I CALCULATE COST FUNCTION, IF IT'S BETTER: UPDATE!
                    z0n,zn,y0n,yn = Net_SA2(x,WEIGHTS_TRIAL-8,
                                tabulated_sigmoid1,minimum_value1,
                                tabulated_sigmoid2,minimum_value2)
     
                    # Cost function with the new configuration.
                    cost_new  = Loss_MSE(t_outs,yn)
                    Loss.append(cost_new)

                    # Saving the best value of the around.
                    if cost_new < loss_old:
                        WEIGHTS_best = copy.deepcopy(WEIGHTS_TRIAL) 
                        WEIGHTS_best[row,column] = w1elenew
                        loss_old = cost_new
                        random_walk = 0
                    if random_walk == 1 and cost_new == LOSS[i-1]:
                        RANDOM_WALK_row.append(row)
                        RANDOM_WALK_col.append(column)
                        NEW_WEIGHT.append(w1elenew)

        # IF THERE ARE MORE THAN ONE MINIMA: RANDOM PICK ONE
        if random_walk==1 and loss_old== LOSS[i-1]:
            WEIGHTS_best = copy.deepcopy(WEIGHTS_TRIAL) 
            rndchoice    = np.random.choice(len(RANDOM_WALK_row))
            WEIGHTS_best[RANDOM_WALK_row[rndchoice],RANDOM_WALK_col[rndchoice]] = NEW_WEIGHT[rndchoice]
        
        # Redefining the next configuration.
        WEIGHTS   = WEIGHTS_best
        z0,z,y0,y = z0n,zn,y0n,yn
        if random_walk==1 : print('rnd_wlk')
        if loss_old == LOSS[i-1] : random_walk=1
        if loss_old  < LOSS[i-1] : random_walk=0
        LOSS.append(loss_old)


# -----------TESTING--------------------------------------------------------#
        # TEST EVERY 10 ITERATIONS
        if i%10==0 or i == N_iter or i==0:
            z0_test, z_test, y0_test, y_test = Net_SA2(X_test, WEIGHTS-8,
                                      tabulated_sigmoid1, minimum_value1,
                                      tabulated_sigmoid2, minimum_value2)

            np.savetxt(path_save + 'y_test{}_{}.csv'.format(str(letter), str(i)),\
                                                     y_test, delimiter=',')
            # orribile
            acc_TEST = (num_samples_test - np.sum(abs(np.round((y_test -\
                               t_outs_test) / np.power(2, N_bit_output),\
                                               0)))) / num_samples_test
            Accuracy_test.append(acc_TEST)

        if False:
            if i % 20 == 0 or i==0:
                plt.plot(y[150:450], 'o', label='Predicted')
                plt.plot(t_outs[150:450], '*', )
                plt.grid(True)
                plt.title('Loss={}'.format(str(loss_old)))
                plt.legend(bbox_to_anchor=(0.95, 0.4), loc=1, borderaxespad=0)
                plt.pause(0.005)
            plt.clf()

    np.savetxt(path_save + 'WEIGHTS_F', WEIGHTS, delimiter=',')
    return y, y_test, Accuracy_test, Loss, LOSS

