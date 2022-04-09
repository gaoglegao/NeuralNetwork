#正向传播

#参数初始化
W1 = W2 = W3 = W4 = W5 = W6 = b1 = b2 = b3 = 0.5

#输入层
A  = 0
E  = 0

#隐藏层
B = A*W1 + E*W2 + b1
C = A*W3 + E*W4 + b2
D = B*W5 + C*W6 + b3
L = 0.005 #学习率


test_data = [
             [[0,0],0], #大于等于2才能输出1
             [[0,1],0],
             [[1,0],1],
             [[1,1],1]
            ] 


def test(data_list,result):
    global A,E,B,C,D
    global W1,W2,W3,W4,W5,W6
    global b1,b2,b3

    A = data_list[0]
    E = data_list[1]

    print("输入 A={0} ,E ={1}".format(A,E))
    print("期望输出值 D={0}".format(result))

    B = A*W1 + E*W2 + b1
    C = A*W3 + E*W4 + b2
    D = B*W5 + C*W6 + b3

    print("实际输出值 D={0}".format(D))


    W1 = W1 - L*2*W5*A*(D - result)
    W2 = W2 - L*2*W5*E*(D - result)
    b1 = b1 - L*2*W5*(D - result)


    W3 = W3 - L*2*W6*A*(D - result)
    W4 = W4 - L*2*W6*E*(D - result)
    b2 = b2 - L*2*W6*(D - result)

    W5 = W5 - L*B*(D - result)
    W6 = W6 - L*C*(D - result)
    b3 = b3 - L*(D - result)

    print("误差值:",abs(D-result))
    print("w1 , w2 , w3 , w4 ,w5, w6 ,b1 ,b2 ,b3 = ",W1,W2,W3,W4,W5,W6, b1,b2,b3)

    return abs(D-result)


qualified_count = 0
while True:
    for item in test_data:
        if test(item[0],item[1]) < 0.000000000001:
            qualified_count+=1
            if qualified_count > 100: #多次训练结果都达到精度要求则完成训练
                print("训练完成")
                exit(0)
        else:
            qualified_count = 0













