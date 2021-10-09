if __name__ == '__main__':
    import numpy as np

    '''Question 1'''
    mu_Sb = 20000
    mu_Sw = 60000
    sigma_Sb = 2000
    sigma_Sw = 3000

    samples = 10000

    x_Sb = np.random.normal(mu_Sb, sigma_Sb, samples)
    x_Sw = np.random.normal(mu_Sw, sigma_Sw, samples)

    success = 0
    failure = 0

    for i in range(len(x_Sb)):
        if x_Sb[i] < 16000 or x_Sb[i] > 24000:
            failure = failure+1
        elif x_Sw[i] < 54000 or x_Sw[i] > 66000:
            failure = failure+1
        elif x_Sb[i]+x_Sw[i] > 100000:
            failure = failure + 1
        else:
            success = success + 1

    pob_success = success/(success+failure)

    print(f'successful samples : {success}\n')
    print(f'failed samples: {failure}\n')
    print(f'probobility of successful requirements given {len(x_Sb)} samples is {pob_success}')

