{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d336a2c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "x = torch.Tensor(5, 3)\n",
    "print(x)\n",
    "y = torch.rand(5, 3)\n",
    "print(y)\n",
    "# let us run the following only if CUDA is available\n",
    "if torch.cuda.is_available():\n",
    "    x = x.cuda()\n",
    "    y = y.cuda()\n",
    "    print(x + y)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
