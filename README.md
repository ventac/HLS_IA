# Embeded-AI

## Developping and improving AI's layers

### Fully connected

File: fc.c

#### First run 

``` bash
Softmax output: 
0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 100.00% 0.00% 0.00% 0.00% %  

Predicted: 6 	 Actual: 6
TOTAL PROCESSING TIME (gettimeofday): 16.000000 s


Errors : 1437 / 10000

Success rate = 85.629997%
```

#### Optimizing it:

To optimize fc.c I've transpformed the expected parameters into constants, this enables vectorization

No improvement in the precision was noticed, but the processing time was reduced by one second and now is: **15.000000 s**




