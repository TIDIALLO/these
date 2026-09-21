# Écrivez un programme qui affiche le triangle suivant pour n = 5 saisi par l'utilisateur : 
 
# 1 
# 1 2 
# 1 2 3 
# 1 2 3 4 
# 1 2 3 4 5 
# correctif :
# n = int(input("Entrez la valeur de n : "))
# for i in range(1, n + 1):
#    print(' '.join(str(x) for x in range(1, i + 1)))
   
#    # version simplifiée :
# n = int(input("Entrez la valeur de n : "))
# for i in range(1, n + 1):
#     for j in range(1, i + 1):
#         print(j, end=' ')
#     print()
    
# print(10//3)

# 1.  Écrivez un programme qui affiche les nombres de 1 à 20, mais remplace : 
# )  les multiples de 3 par  "Fizz" 
# )  les multiples de 5 par  "Buzz" 
# )  les multiples de 3 et de 5 par  "FizzBuzz" 
#       Exemple : 1  2  Fizz  4  Buzz  Fizz  7  8  Fizz  Buzz  11  Fizz  13  14  FizzBuzz ...

# for i in range(1, 21):
#    if i % 3 == 0 and i % 5 == 0:
#       print("FizzBuzz", end=' ')
#    elif i % 3 == 0:
#       print("Fizz", end=' ')
#    elif i % 5 == 0:
#       print("Buzz", end=' ')
#    else:
#       print(i, end=' ')

total = 0 
for i in range(1, 7): 
    if i % 3 == 0: 
        total += i * 3 
    else: 
        total += 1 
print(total) 
 
# Écrivez un programme qui demande une phrase à l'utilisateur et compte séparément : le nombre de lettres 
#majuscules, le nombre de lettres minuscules, le nombre de chiffres et le nombre d'espaces.   [2 pts] 
 #      Exemple : "Bonjour 2025" -> 1 majuscule, 6 minuscules, 4 chiffres, 1 espace
phrase = input("Entrez une phrase : ")
majuscule = 0
minuscule = 0
chiffre = 0
espace = 0
for char in phrase:
    if char.isupper():
        majuscule += 1
    elif char.islower():
        minuscule += 1
    elif char.isdigit():
        chiffre += 1
    elif char.isspace():
        espace += 1
print(f"{majuscule} majuscule(s), {minuscule} minuscule(s), {chiffre} chiffre(s), {espace} espace(s)")   

#  Écrivez un programme qui affiche les multiples de 9 entre 1 et 100 sur une seule ligne, séparés par des 
#espaces. Affichez aussi leur nombre total et leur somme.
total = 0
count = 0
for i in range(1, 101):
    if i % 9 == 0:
        print(i, end=' ')
        total += i
        count += 1
print(f"\nNombre total : {count}, Somme : {total}")

#  Écrivez une fonction compter_mots(phrase) qui retourne le nombre de mots d'une phrase sans utiliser 
#len(phrase.split()). Utilisez une boucle pour parcourir les caractères.   [2 pts] 
#      Exemple : compter_mots("Bonjour tout le monde") retourne 4. 
#Testez avec : "Python est super", "  espaces   multiples  ", "un". 
def compter_mots(phrase):
    count = 0
    in_word = False
    for char in phrase:
        if char.isspace():
            in_word = False
        else:
            if not in_word:
                count += 1
                in_word = True
    return count
 
#2.  Ecrivez un programme qui demande le prix d'un article en FCFA et affiche la catégorie :   [3 pts] 
#)  "Gratuit" si le prix est égal à 0 
#)  "Très accessible" si le prix est entre 1 et 1000 FCFA 
#)  "Accessible" si le prix est entre 1001 et 5000 FCFA 
#)  "Cher" si le prix est entre 5001 et 20000 FCFA 
#)  "Luxe" si le prix dépasse 20000 FCFA 
#   Affichez aussi le prix après remise de 10% si le prix dépasse 5000 FCFA, 
#       ou le prix après majoration de 5% si le prix est inférieur ou égal à 1000 FCFA. 
prix = float(input("Entrez le prix de l'article en FCFA : "))
if prix == 0:
   categorie = "Gratuit"
elif 1 <= prix <= 1000:
   categorie = "Très accessible"
   prix *= 1.05  # majoration de 5%
elif 1001 <= prix <= 5000:
   categorie = "Accessible"
elif 5001 <= prix <= 20000:
   categorie = "Cher"
   prix *= 0.9  # remise de 10%
else:
   categorie = "Luxe"
   prix *= 0.9  # remise de 10%
   

#version simplifiée :
prix = float(input("Entrez le prix de l'article en FCFA : "))
if prix == 0:
    categorie = "Gratuit"
elif prix <= 1000:
    categorie = "Très accessible"
    prix *= 1.05  # majoration de 5%
elif prix <= 5000:
    categorie = "Accessible"
elif prix <= 20000:
    categorie = "Cher"
    prix *= 0.9  # remise de 10%
else:
    categorie = "Luxe"
    prix *= 0.9  # remise de 10%
    
# 2.  Ecrivez une fonction est_valide(mot_de_passe) qui retourne True si le mot de passe respecte toutes ces 
# règles, False sinon :   [2 pts] 
#)  Au moins 8 caractères 
#)  Contient au moins un chiffre 
#)  Contient au moins une lettre majuscule 
#       Testez avec : "Python2025", "faible", "SANSCHI", "Bon1", "Assez_Long9A". 
def est_valide(mot_de_passe):
    if len(mot_de_passe) < 8:
        return False
    has_digit = any(char.isdigit() for char in mot_de_passe)
    has_upper = any(char.isupper() for char in mot_de_passe)
    return has_digit and has_upper
#version simplifiée :
def est_valide(mot_de_passe):
      return len(mot_de_passe) >= 8 and any(char.isdigit() for char in mot_de_passe) and any(char.isupper() for char in mot_de_passe)
   
#version simplifiée avec condition simplifiée :
def est_valide(mot_de_passe):
    return len(mot_de_passe) >= 8 and any(char.isdigit() for char in mot_de_passe) and any(char.isupper() for char in mot_de_passe)
 
# version debutant :
def est_valide(mot_de_passe):
    if len(mot_de_passe) < 8:
        return False
    has_digit = False
    has_upper = False
    for char in mot_de_passe:
        if char.isdigit():
            has_digit = True
        if char.isupper():
            has_upper = True
    return has_digit and has_upper
 
#veresion simple et versifie chaque condition :
def est_valide0(mot_de_passe):
    if len(mot_de_passe) < 8:
        return False

    contient_chiffre = False
    contient_majuscule = False

    for caractere in mot_de_passe:
        if caractere.isdigit():
            contient_chiffre = True

        if caractere.isupper():
            contient_majuscule = True

    if contient_chiffre and contient_majuscule:
        return True
    else:
        return False

print(est_valide0("Python2025"))  # True