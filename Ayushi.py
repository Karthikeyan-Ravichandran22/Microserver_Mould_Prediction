import datetime

def input_date_of_birth():
    while True:
        dob_str = input("Enter your date of birth (dd/mm/yyyy): ")
        try:
            dob = datetime.datetime.strptime(dob_str, '%d/%m/%Y').date()
            if dob > datetime.date.today():
                print("Date of birth cannot be in the future. Please try again.")
                continue
            return dob
        except ValueError:
            print("Invalid format or date. Please enter a valid date in the format dd/mm/yyyy.")

def calculate_age(dob):
    today = datetime.date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age


def check_birthday(dob):
    today = datetime.date.today()
    return today.month == dob.month and today.day == dob.day

def main():
    dob = input_date_of_birth()
    age = calculate_age(dob)
    print(f"You are {age} years old.")

    if check_birthday(dob):
        print("Happy Birthday!")

if __name__ == "__main__":
    main()



