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
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            total = 0
            for k in range(size):
                total += A[i][k] * B[k][j]
            row.append(total)
        result.append(row)
    return result

def matrix_power(A, m):
    size = len(A)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]  
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



#Q4
def occurance(s):
    max=0
    ch=''
    for i in range(len(s)):
        c=0
        for j in range(i,len(s)):
            if (s[i]==s[j]):
                c+=1
        if(c>max):
            max=c
            ch=s[i]
    return ch,max

def main():
    s=input("Enter a string")
    ch,max=occurance(s)
    print(f"{ch} is the most occured character and count is {max}")

main()

#Q5
import random

def calculate_mean(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def calculate_median(numbers):
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 1:
        return sorted_nums[n // 2]
    else:
        mid1 = sorted_nums[n // 2 - 1]
        mid2 = sorted_nums[n // 2]
        return (mid1 + mid2) / 2

def calculate_mode(numbers):
    frequency = {}
    for num in numbers:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1

    max_freq = max(frequency.values())
    mode_list = []
    for num, freq in frequency.items():
        if freq == max_freq:
            mode_list.append(num)

    return mode_list

def main():
    numbers = []
    for _ in range(25):
        numbers.append(random.randint(1, 10))

    print("Mean:", calculate_mean(numbers))
    print("Median:", calculate_median(numbers))
    print("Mode:", calculate_mode(numbers))

main()


