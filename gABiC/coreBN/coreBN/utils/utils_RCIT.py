def pgamma(q,shape,rate):
    """
    Calculates the cumulative of the Gamma-distribution.
    Generates probability density values in vector form , 
    cumulative probability density values and moment about zero values for Gamma Distribution bounded between [0,1].
    """
    from scipy.stats import gamma    
    result=gamma.cdf(x=q,a=shape,loc=0,scale=rate)
    return result

def alls(vec):
    return all(x>0 for x in vec)

def checkCoeffsArePositiveError(coeff):
    if(len(coeff) == 0):
        return True
    
    if(not alls(coeff)):
        return True
    return False

def getCoeffError(coeff):
    if(len(coeff) == 0):
        return "empty coefficient vector."

    if(not alls(coeff)):
        return "not all coefficients > 0."
    
    return "unknown error."

def checkXvaluesArePositiveError(x):
    if(len(x) == 0):
        return True
    
    if(not alls(x)):
        return True
    
    return False

def getXvaluesError(x):
    if(len(x) == 0):
        return "empty x vector."

    if(not alls(x)):
        return "not all x-values > 0."

    return "unknown error."