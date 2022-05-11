# concatenation
from itertools import chain, combinations
from re import findall, M
from itertools import combinations_with_replacement
from itertools import chain
from math import ceil
from math import floor, sqrt
from copy import copy
from re import finditer
from itertools import combinations
from functools import reduce
num = input().split(',')
pos_5, pos_8 = num.index('5'), num.index('8')
num1 = reduce(lambda x, y: int(x) + int(y),
              num[0: pos_5] + num[pos_8 + 1: len(num)])
num2 = int(''.join(num[pos_5:pos_8 + 1]))
print('{}\n{}'.format(num1, num2))

# distinct subarray
n1 = int(input())
n2 = int(input())
arr = [i for i in range(n1, n2 + 1)]
_arr = [arr[i:j + 1] for i in range(len(arr)) for j in range(i, len(arr))]
_arr = list(filter(lambda sub_arr: sum(sub_arr) % 2 != 0, _arr))
print(len(_arr))

# palindrome
num = input()
_len = int((len(num)) / 2)
left_str = num[:_len][::-1]
x, y = num[:_len + 1] + left_str, num[:_len] + \
    chr(ord(num[_len]) + 1) + left_str
diff_x, diff_y = int(num) - int(x), int(num) - int(y)
max_diff = max(diff_x, diff_y)
if(max_diff == diff_x and max_diff < 0):
    print(int(x))
else:
    print(int(y))

# parenthesis matching
op, cls, acc_arr = ['{', '(', '['], ['}', ')', ']'], []
instr = input()
_len = len(instr)

if(instr[0] in op):
    acc_arr.append(1)
elif(instr[0] in cls):
    acc_arr.append(-1)
else:
    acc_arr.append(0)
# acumulator
for i in range(1, _len):
    app = acc_arr[i - 1] + 1 if instr[i] in op else acc_arr[i - 1] - \
        1 if instr[i] in cls else acc_arr[i - 1]
    acc_arr.append(app)

last = acc_arr[_len - 1]
if(last > 0):
    print(last + _len)
elif(last < 0):
    for i in range(0, _len):
        if(acc_arr[i] < 0):
            print(i + 1)
            break
else:
    print(0)

# sum of factor
num = list(map(int, input().split(',')))
sum_arr = set()
for i in range(0, len(num)):
    factors = []
    if(num[i] == 1):
        factors.append(1)
    else:
        for j in range(1, num[i]):
            if(num[i] % j == 0):
                factors.append(j)
    _sum = sum(factors)
    if(_sum in num and _sum != 0):
        sum_arr.add(_sum)

print(list(sum_arr))

# Maximum subarray


def isfibonacci(arr: list):
    if(arr.__len__() < 3):
        return False
    for i in range(2, arr.__len__()):
        if (arr[i] != arr[i - 1] + arr[i - 2]):
            return False
    return True


array = list(map(int, input().split(',')))

sub_arrays = []
for i in range(1, array.__len__() + 1):
    for j in map(list, combinations(array, i)):
        j.sort()
        sub_arrays.append(j)

print(list(filter(isfibonacci, sub_arrays)))

# Special string reverse
string = input()
finds = []
for found in (finditer('@', string)):
    finds.append(found.span()[0])
unspec_str = ''.join(filter(lambda x: x not in '@', reversed(string)))
for found in finds:
    unspec_str = unspec_str[:found] + '@' + unspec_str[found:]

print(unspec_str)

# Special character out
rsult = []
odd = []
pending = False
instr = input()
for i, c in enumerate(instr, 0):
    if(ord(c) in range(ord('0'), ord('9') + 1)):
        if(int(c) % 2 != 0):
            pending = True
            odd.insert(0, c)
        else:
            if (pending == True):
                rsult.append(odd.pop())
                if(len(odd) == 0):
                    pending = False
                rsult.append(c)
            elif(len(odd) != 0):
                rsult.append(c)
                pending = True
rsult.append(odd.pop())

print(rsult)
print(odd)

# Password generation
# Robert: 36787,Tina: 68721,Jo:56389
instr = input().split(',')
password = str()
indict = dict([_instr.split(':') for _instr in instr])
for name in indict:
    _len = len(name)
    if(str(_len) not in indict[name]):
        while(_len > -1):
            if(str(_len) in indict[name]):
                break
            _len -= 1
        if(_len > -1):
            password += name[_len - 1]
        else:
            password += 'X'
    else:
        password += name[_len - 1]

print(password)

# Even number
string = input()
num = list()
index = dict()
for char in string:
    if (ord(char) in range(ord('0'), ord('9') + 1)):
        num.append(int(char))
        if(int(char) % 2 == 0):
            index[int(char)] = len(num) - 1
_index = index[min(index)]
if(index != -1):
    num.append(num[_index])
    num.__delitem__(_index)
_num = copy(num[:-1])
_num.sort(reverse=True)
_num.append(num[len(num) - 1])
print(reduce(lambda x, y: (x * 10) + y, _num))

# Matrix problem
row = int(input())
mat = []
# insert all value of each row
for i in range(row):
    mat.append(list(map(int, input().split())))
    print(mat)
    col = len(mat[0])
    out = []
for r in range(row):
    for c in range(col - 2):
        if mat[r][c] == mat[r][c + 1] == mat[r][c + 2]:
            out.append(mat[r][c])

for r in range(row - 2):
    for c in range(col):
        if mat[r][c] == mat[r + 1][c] == mat[r + 2][c]:
            out.append(mat[r][c])

for r in range(row - 2):
    for c in range(col - 2):
        if mat[r][c] == mat[r + 1][c + 1] == mat[r + 2][c + 2]:
            out.append(mat[r][c])
print(out)
print(min(out))

# String rotation
instr = input().split(',')
indict = dict([_str.split(':') for _str in instr])
sum_even = lambda num: sum([pow(int(x), 2)for x in num]) % 2 == 0
for key in indict.keys():
    if(sum_even(indict[key])):
        for i in range(0, 1):
            key = key[-1:] + key[:-1]
        print(key)
    else:
        for i in range(0, 3):
            key = key[-1:] + key[:-1]
        print(key)

# Pronic number
num = input()


def is_pronic(_num):
    k = floor(sqrt(_num))
    return True if (k * (k + 1) == _num) else False


result = list()
subdigit = 0
flag = False
for j in range(0, len(num)):
    for i in range(j + 1, len(num)):
        subdigit = int(num[j:i])
        if(is_pronic(subdigit)):
            result.append(subdigit)
            break
        if(i >= len(num) - 1):
            flag = True
            break
    if(flag == True):
        break
print(result)

# Longest palindrome
s = input()
substrings = [s[i:j + 1] for i in range(len(s)) for j in range(i, len(s))]
_substrings = filter(lambda _str: len(_str) > 1 and ''.join(
    reversed(_str)) == _str, substrings)
print(set(_substrings))

# Otp generation
result = list()
_c = int()
for c in input():
    _c = int(c)
    if(_c % 2 != 0):
        result.append(str(_c ** 2))
print(''.join(result))

# unique substrings
instr = input()
substr = list(filter(lambda string: len(string) ==
              len(set(string)), [[x for x in combinations(instr, i)] for i in range(3, len(instr))]))
max_len = max(map(lambda x: len(x), substr))
for i in range(len(substr)):
    if (len(substr[i]) == max_len):
        print(substr[i])
        break
    if(i == len(substr)):
        print(-1)

# Coding Task 1:
arr = input().split(' ')
h = int(ceil(len(arr) / 2))
print(arr[h:] + arr[:h])

# Coding Task 2:
instr = input("String : ")
finds = [finditer(substr, instr)
         for substr in input("Substrings : ").split(' ')]
for matches in finds:
    for match in matches:
        instr = instr[:match.span()[0]] + '\0' * (match.span()
                                                  [1] - match.span()[0]) + instr[match.span()[1]:]
print(instr)

# First program
pwd = list()
instr = input()
for i in range(len(instr)):
    if(i % 2 != 0):
        pwd.append(pow(int(instr[i]), 2))
print(pwd[:-1])
print(reduce(lambda a, b: b + (a * 10), pwd))

# Second program
spec = '#&'
instr = input()
outstr = [' '] * instr.__len__()

for find in finditer('|'.join(spec), instr):
    _find = find.span()[0]
    outstr[_find] = instr[_find]

_instr = list(filter(lambda c: c not in spec, instr))

for i in range(0, len(outstr)):
    if(outstr[i] != ' '):
        continue
    outstr[i] = _instr.pop()

print(''.join(outstr))

# Maximum even number
instr = input()
numbers = list(set(map(int, filter(lambda c: ord(
    c) in range(ord('0'), ord('9') + 1), instr))))
numbers.sort(reverse=True)

flag = False
if(int(numbers[-1]) % 2 == 0):
    for i in range(len(numbers), -1):
        if(int(numbers[i]) % 2 == 0):
            flag = True
            numbers = numbers[:i] + numbers[i + 1:] + numbers[i]
            break
if(flag == True):
    numbers = -1
print(numbers)

# Sum of factors


def find_factors(n: int):
    if(n > 1):
        return list(filter(lambda y: (n % y) == 0, range(2, n)))
    return [n]


nums = map(int, input().split(','))
factors = list(chain(*map(find_factors, nums)))
if(sum(factors) in nums):
    factors.sort()
print(factors)

# Prefix and suffix
instr = input()
index = -1
for i in range(int(len(instr) / 2)):
    if(instr[i] == instr[-i]):
        index += 1
    else:
        break
print(-1) if index == -1 else print(index + 1)

# Palindrome int
_num = int(input())
__num = _num
num = 0
while(_num >= 1):
    num = int((num * 10) + _num % 10)
    _num /= 10
print(num == __num)

# Sum of Natural numbers in binary form
print(sum([int(bin(x)[2:]) for x in range(int(input()) + 1)]))


# Special Prime


def sum_of_digits(n: int):
    result = 0
    while(n >= 1):
        result = result + (n % 10)
        n = int(n / 10)
    return result


is_prime = lambda n: ((n - 1) % 6 == 0 or (n + 1) %
                      6 == 0 or n in (2, 3)) and n > 1
num = int(input())
_range = list(range(num + 1))
f_range = map(sum_of_digits, _range)
print(list(filter(lambda T: is_prime(
    T[0] + T[1]), combinations_with_replacement(f_range, 2))).__len__())

# Break the waffle


class h_cut:
    def __init__(self, cost):
        self.cost = cost
        self.o_cost = cost

    def __gt__(self, other):
        if(self.cost < other.cost):
            return True
        else:
            return False


class v_cut:
    def __init__(self, cost):
        self.cost = cost
        self.o_cost = cost

    def __gt__(self, other):
        if(self.cost < other.cost):
            return True
        else:
            return False


(m, n) = map(int, input().split(' '))
h_cost = list(map(lambda x: h_cut(int(x)), input().split(' ')))[:m]
v_cost = list(map(lambda x: v_cut(int(x)), input().split(' ')))[:n]
Cost = h_cost + v_cost
Cost.sort()

for i in range(len(Cost)):
    if(type(Cost[i]) == h_cut):
        for j in range(i + 1, len(Cost)):
            if(type(Cost[j]) == v_cut):
                Cost[j].cost += Cost[j].o_cost
    else:
        for j in range(i + 1, len(Cost)):
            if(type(Cost[j]) == h_cut):
                Cost[j].cost += Cost[j].o_cost
print(sum([x.cost for x in Cost]))

print(findall(r'[a-z0-9:]*\/[a-z0-9:]+\\[a-z]+', input(), M))

# Uniformity
char_map = dict()
string = input()
limit = int(input())
for char in set(string):
    char_map[char] = string.count(char)
print(char_map[list(char_map.keys())[0]] + (limit % len(string)))

# Output-Longest XSeries
arr = [2, 6, 3, 5, 8, 9]
n = len(arr)
result = list()
for i in range(len(arr) - 1):
    for j in range(i + 1, len(arr) - 1):
        if(arr[i] + arr[j] == arr[j + 1]):
            result.append([arr[i], arr[j], arr[i] + arr[j]])
print(result)

# longest substring of unique characters
instr = input()
combinations = [instr[i:j + 1]
                for i in range(len(instr)) for j in range(i, len(instr))]
print(max(*list(filter(lambda x: len(set(x)) == len(x), combinations))))

# 4 digit OTP
n = input()
result = '0'
for i in range(len(n)):
    if(i % 2 != 0):
        result += str(pow(int(n[i]), 2))
print(int(result[:5]))

# Reversed Special character string
instr = input()
special = '&#'
result = [None] * len(instr)
s_index = 0
for i in range(len(instr) - 1, -1, -1):
    if(instr[i] in special):
        result[i] = instr[i]
for i in range(len(instr) - 1, -1, -1):
    if(instr[i] not in special):
        while(result[s_index] != None):
            s_index += 1
        result[s_index] = instr[i]
print(''.join(result))


# Sum of factors
def get_factors(n: int):
    if(n <= 1):
        return [n]
    factors = list()
    for i in range(1, n):
        if(n % i == 0):
            factors.append(i)
    return factors


_in = list(map(int, input().split()))
arr = [sum(get_factors(i)) for i in _in]
print(arr)
filtered = list(filter(lambda x: x != None, [
                x if x in _in else None for x in arr]))
print(filtered)
print(-1) if filtered == [] else print(set(filtered))


# Largest even number in string
def findeven(s: str):
    _s = sorted(set(s), reverse=True)
    if(int(_s[-1]) % 2 == 0):
        return _s
    for i in range(len(_s) - 1, -1, -1):
        if(int(_s[i]) % 2 == 0):
            _s.append(_s[i])
            _s = _s[:i] + _s[i + 1:]
            break
    return _s


S = ''.join(filter(lambda c: str(c).isdigit(), input()))
print(findeven(S))

# Make palindrome


def check_palindrome(instr: str):
    if(instr[::-1] == instr):
        print(len(instr))
    else:
        _instr = str(int(instr) + int(instr[::-1]))
        print(_instr)
        check_palindrome(_instr)


_instr = input()
check_palindrome(_instr)

# Prefix also a suffix
instr = input()
length = -1
for i in range(int(len(instr) / 2)):
    if(instr[:i] == instr[-i:]):
        length = i
        break
print(length)
for i in range(length, int(len(instr) / 2)):
    if(instr[:i] != instr[-i:]):
        length = i
        break
print(instr[:length])

# Find substring based on conditions
instr = input()

filter_conditions = lambda s: len(set(s)) == len(s) and len(s) >= 3

combinations = [instr[i:j + 1]
                for i in range(len(instr)) for j in range(i, len(instr))]
_combinations = list(filter(filter_conditions, sorted(
    combinations, reverse=True, key=len)))
print(-1) if len(_combinations) == 0 else print(_combinations[0])

# Find combinations of integers with given conditions (fibonacci)


def filter_conditions(n):
    if(len(n) < 4):
        return False
    for i in range(2, len(n)):
        if(n[i - 1] + n[i - 2] != n[i]):
            return False
    return True


nums = list(map(int, input().split()))
combos = list(
    map(sorted, chain(*[combinations(nums, i) for i in range(2, len(nums))])))
_combos = sorted(filter(filter_conditions, combos), key=len, reverse=True)
print(-1) if len(_combos) == 0 else print(_combos)
