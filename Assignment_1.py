# ques 1

class DivisionByZeroError(Exception):
    """Exception raised for division by zero."""
    pass
def divide(a, b):
    if b == 0:
        raise DivisionByZeroError("Cannot divide by zero.")
    return a / b
# Example usage
try:
    result = divide(10, 0)
except DivisionByZeroError as e:
    print(f"Error: {e}")

# output==>Error: Cannot divide by zero.

# ques 2
# we have to make 2d matrix
#   x1   x2  x3
#   10  20  30 
#   14  24  34
#   16  26  36


# we have to store data in txt as 
# x1, x2, x3
# 10, 20, 30
# 14, 24, 34 
# 16, 26, 36

arr = [['x1','x2', 'x3'],
       [10, 20, 30],
       [14, 24, 34],
       [16, 26, 36]
]

# write
with open("assignment1.csv",'w') as file:
    for i in arr:
        for j in i:
            if(i.index(j)==len(i)-1):
                file.write(str(j))
            else:
                file.write(str(j)+', ')
        file.write('\n')

# read
file=open("assignment1.csv")
for line in file.readlines():
    print(line,end='')

# x1, x2, x3
# 10, 20, 30
# 14, 24, 34
# 16, 26, 36