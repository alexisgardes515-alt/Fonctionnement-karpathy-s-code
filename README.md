# Fonctionnement du code microGPT de Andrej Karpathy 

L'objectif de ce code est de créer, à partir d'une base de prénoms, de nouveaux prénoms innexistants.
Pour cela A.Karpathy a écrit un code en python pur de seulement **~200 lignes** à l'aide des bibliothèques de bases de python.
Ce code se décompose en **6 parties** :

## 0-Importation des bibliothèques et du dataset (depuis github)

```python
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos



# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

## 1-Le Tokenizer

Le Tokenizer permet de transformer une chaîne de caractère en Tokens (suite d'entiers). Les Tokens sont les éléments de base qui nous permettrons de constituer les prénoms (il y en aura 26, 1 par lettre, +1 pour le BOS, un token spécial utilisé seulement pour marquer le début d'une séquence <=> un prénom )

```python
# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens
print(f"vocab size: {vocab_size}")
```

## 2-L'autoguard de microGPT : une reprie simplifiée du mécanisme de l'autoguard PyTorch

*Autoguard* : c'est un mécanisme permettant à un réseau de neurones de comprendre ses erreurs et de les corriger

Les étapes de **l'Autoguard** :

  i-*Forward pass* (l'Aller) :
  Cette étape commence dès le départ et permet d'enregistrer chaque opération dans un graphe de calcul.

  ii- *Backward pass* (le Retour) : 
  Cette étape s'exécute après le transformer et permet de remonter le graphe pour calculer le gradient de chaque poids (avec    la règle de la chaîne).

```python
# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

# 3-Création d'un vecteur pour identifier un token et sa position : WTE + WPE

Lorsque l'on veut analyser un token, il n'est pas directement exploitable car il est sous la forme d'un identifiant (par exemple : pour "a" tk_id=0 , pour "b" tk_id=1 ....) 
L'objectif est donc de créer un vecteur de dimension 16 pour être exploité par le transformer.

(La dimension 16 est un choix arbitraire de A.Karpathy car il est adapté au modèle microGPT et un dataset d'environ 32 000 prénoms.
A titre de comparaison, pour le modèle GPT-2-small la dimension est de 768 car le dataset correspond à internet tout entier,
pour le modèle GPT-3 la dimension est de 12 288 car le dataset correspond à internet x 10,)

  i- WTE ( Word Token Embedding) : Transforme l'id du token en vecteur de dimension 16

  ii- WPE (Word Position Embedding) : Transforme l'id de la position du token en vecteur de dimension 16

```python
# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16    # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]    # Initialisation de la matrice avec les poids non entraînés 
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
```

Enfin on somme les deux vecteurs pour obtenir un vecteur X contenant les deux informations.

```python
tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection
```  

## 4-


































