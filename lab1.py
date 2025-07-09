#Q1
def pairs(a):
    sum=0
    for i in range (0,6):
        for j in range (0,6):
            if(a[i]+a[j]==10):
                sum+=1
    return int(sum/2)

def main():
    a=[2,7,4,1,3,6]
    print(pairs(a))

main()

#Q2
def main():
    user_inp=input("Enter a list of elements separated by space").split()
    user_inp = list(map(int, user_inp)) #cauase list comes as string after split so map int to list
    if(len(user_inp)<3):
        print("Range determination not possible")
    else:
        print(calcrange(user_inp))

def calcrange(user_inp):
    small=int(min(user_inp))
    most=int(max(user_inp))
    return int(most-small)

main()


#Q3
def matrix_multiply(A, B):
    size = len(A)
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_power(A, m):
    size = len(A)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]  # Identity matrix
    for j in range(m):
        result = matrix_multiply(result, A)
    return result


def main():
    A=[]
    size=int(input("Enter size of matrix"))
    for i in range(size):
        row = []
        for j in range(size):
            val = int(input(f"Enter value for A[{i}][{j}]: "))
            row.append(val)
        A.append(row)

    m=int(input("enter m"))
    result=matrix_power(A,m)
    for row in result:
        print(row)

main()


