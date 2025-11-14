import bcrypt
passwords = ["YourStrongPwd1", "AnotherPwd2"]
for p in passwords:
    print(bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode())
