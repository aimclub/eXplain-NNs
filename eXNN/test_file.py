import random

def sum(x: int, y: int):
    return x + y

if __name__ == "__main__":
    res = sum(random.randint(0, 100), random.randint(0, 100))
    print(res)
    
