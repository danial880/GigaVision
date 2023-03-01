def find_most_frequent(arr):    
    frequency = {}
    for num in arr:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    max_frequency = max(frequency.values())   
    most_frequent = [key for key, value in frequency.items() if value == max_frequency]   
    return max(most_frequent)


a = [1,2,2,2,2,5,5,5,9,9,9,100,100,100,100,100,100]
print(find_most_frequent(a))


def replace_morse_code(morse_code):
    morse_codes = []
    i = 0
    while i < len(morse_code) - 1:
        if morse_code[i:i+2] == "..":
            new_morse_code = morse_code[:i] + "--" + morse_code[i+2:]   
            morse_codes += replace_morse_code(new_morse_code)
            morse_codes.append(new_morse_code)
        i += 1
    if not morse_codes:
        morse_codes.append(morse_code)
    return list(set(morse_codes))


#code = input()
#morse_code = replace_morse_code(code)
#print(morse_code)

def factorial(num):
    if num == 1:
        return 1
    else:
        fact = num * factorial(num-1)
        return fact
fact = factorial(5)
print(fact)
