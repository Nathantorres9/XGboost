import numpy as np
data = np.loadtxt('E:/FIR_Implementation/txt/X_train_processed.txt')
def float2fix_point(data, exp, gain, size):
    # '''
    # :param data: 信号源数据
    # :param exp:  浮点数转定点数的位宽
    # :param gain: 浮点数整体乘以增益，增益为power(2,15)
    # :param size: 转换多少点数
    # :return:
    # '''
    if size > len(data):
        print("error, size > len(data)")
        return
    data = [int(np.floor(data[i] * np.power(2, gain) )) for i in range(size)]
    fmt = '{{:0>{}b}}'.format(exp)
    n = np.power(2, exp)
    for i in range(size):
        if data[i] > (n //2  - 1):
           print("error")

        if data[i] < 0:
            d = n + data[i]
        else:
            d = data[i]
        data[i] = fmt.format(d)
    # data = [bin(data[i]) for i in range(4096)]
    np.savetxt('E:/FIR_Implementation/cos/X_train_binary.txt', data, fmt='%s')
float2fix_point(data, 16, 6, 16000000)