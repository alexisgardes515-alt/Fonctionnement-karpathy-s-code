# Fonctionnement du code microGPT de Andrej Karpathy 

L'objectif de ce code est de créer, à partir d'une base de prénoms, de nouveaux prénoms innexistants.
Pour cela A.Karpathy, en se basant sur une architecture GPT (Generative Pre-trained Transformer), a écrit un code en python pur de seulement **~200 lignes** à l'aide des bibliothèques de bases de python.
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

## 2-L'autoguard de microGPT : une reprise simplifiée du mécanisme de l'autoguard PyTorch

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

## 4- Le GPT: le coeur du modèle

Il prend en entrée notre vecteur X afin d'en resortir un nombre de probabilité égale au nombre de token. Chaque probabilité associé au token T correspond au pourcentage de chance que le token suivant soit le token T.
Le GPT est constitué de 2 parties :

i-Le Transformer est constitué d'un enchaînement de 4 blocs et de 2 connexions residuelles: 
    
1) RMSNorm (Root Mean Square Norm) : Il permet de mettre chaque valeur à une échelle proche de 1 pour éviter les valeurs trop basses ou trop élévée

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```
2) L'attention : Se base sur trois variables : q (la requête), k (la clé), v (la valeur). D'abord on effectue le produit scalaire entre Q (du token actuel) et K (des tokens passés). Le résultat obtenu est un score qui plus il est élevé, plus ce token est pertinent. Ensuite on le normalise en pourcentage avec softmax(). Enfin on multiplie les poids de softmax() par les Valeurs et on ajoute le résultat à notre token pour qu'il enregistre ce que les tokens précédent lui ont "appris".

         ```python
        for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))] 
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
         ```
On effectue ensuite une connexion résiduelle pour additioner le token original et le résultat de l'attention.

```python
x = [a + b for a, b in zip(x, x_residual)]
```

4) Une autre utilisation de RMSNorm pour remettre à l'échelle
   
5) Le MLP (Multi-Layer Perceptron): Après avoir obtenu de nombreuses informations d'apprentissage via l'attention, le MLP va permettre de les transformer en des données exploitables. Pour ce faire on agrandit les dimensions avec la fonction *.mlp_fc1* de notre vecteur (ici on passe de 16 à 64) puis on utilise la fonction *.relu()* pour mettre à 0 toutes les valeurs négatives de notre vecteur, enfin on repasse en dimension 16 avec la fonction *.mlp_fc2*. La fonction relu permet ici d'obtenir un modèle non-linéaire et non simplifié.

```python
x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
```
On effectue ensuite une connexion résiduelle pour additioner le token avant application du MLP et le résultat du MLP.
```python
x = [a + b for a, b in zip(x, x_residual)]
```
ii- Génération des probabilités : 

Pour terminer on va générer nos probabilités d'obtenir chaque charactère après le passage dans le transformer 

```python
logits = linear(x, state_dict['lm_head'])
return logits
```








## 5-Boucle d'entraînement : la phase d'apprentissage

On crée une boucle qui répète un nombre arbitraire de fois (ici 1000) les étapes précédentes afin d'entraîner notre modèle :

1) On prend un prénom de notre dataset que l'on convertit en token

```python
 doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
```

2) Prédiction du prochain token et mesure de la pertinance des probabilités (placée dans une variable appelée **loss** ). Plus la perte est élevée et moins les probabilités seront pertinantes.

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.
```

3) On applique ensuite le *backward()* pass de notre autoguard afin de **calculer le gradient** des calculs enregistrés depuis le début du proccesus.

```python
loss.backward()
```

4) Mise à jour des poids avec l'optimisation **Adam** (Adaptative Moment Estimation ):

Adam est une méthode d'optimisation se basant sur l'ajout de 2 variables pour tracer une moyenne pondérée des gradients du *backwardpass* où les gradients les plus récents comptent plus que les anciens.
Le premier est le momentum (m): il permet l'accélaration (resp. décélération) de la vitesse d'apprentissage lorsque les gradients sont dans la même direction (resp. dans des directions opposées), car cela signifie que les paramètres sont cohérents (resp. incohérent) entre eux.
Le deuxième est l'adaptation du taux d'apprentissage (v): il permet d'adapter la vitesse d'apprentissage en fonction de la valeur du gradient: plus un gradient est grand, plus le taux d'apprentissage diminue et inversemment.
## 6-L'Inférence : la génération des nouveaux prénoms 

C'est l'étape de l'obtention des résultats. Pour former un nouveau prénom le script procède de la manière suivante: On prend notre **token de départ** (le BOS), on calcule à l'aide du transformer les différentes **probabilités** pour savoir quelle lettre a le plus de chance de venir après, puis on l'**ajoute à notre prénom**.

On répète la boucle **probabilités** -> **ajout de lettre** jusqu'à retomber sur le token BOS, symbolisant le début d'un nouveau prénom

```python
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```


























