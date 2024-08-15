import PowerNetwork
from PowerNetwork import Network
import pypsa
import numpy as np
import pandas as pd



##file = "./Data/IEEE_14bus.xlsx" ##PATH FOR SYSTEM DATA
file = "./Data/testbus.xlsx" ##PATH FOR SYSTEM DATA


## LET'S INITIALIZE THE POWER NETWORK USING ABOVE DATA

nw = Network(file)
nbus = nw.busCount
print(f"\n\n NETWORK INITIALIZED: \n {nbus} buses \n {nw.lineCount} lines \n {nw.generatorCount} generators \n {nw.generatorCount} transformers and \n {nw.shuntcapacitorCount} shunt capacitors \n")
print(f"BUS PARAMETERS \n {nw.buses}") ## 9 (Bus no., Voltage magnitude p.u., Voltage phase angle, Generation MW, Generation MVAR, Load MW, Load MVAR, Reactive Power Limits Qmin, Reactive Power Limits, Qmax)
print(f"\n LINE PARAMETERS \n {nw.lines}") ## 7(Line no., From bus, To bus, Line Impedance p.u. Resistance, Line Impedance p.u. Reactance, Half line charging susceptance p.u., MVA rating)
print(f"\n GENERATOR PARAMETERS \n {nw.generators}") ## 6(Generator no., Pmin, Pmax, $/MWhr2, $/MWhr, $/hr)
print(f"\n TAP POSITION PARAMETERS \n {nw.tapPositions}") ##3(From bus, To bus, Tap setting p.u.)
print(f"\n SHUNT CAPACITOR PARAMETERS \n {nw.shuntCapacitors}") ##2(Bus number, Susceptance p.u.)


## NOW LET'S DO THE POWER FLOW ANALYSIS(NEWTON-RAPHSON ITERATION)


"""
BASIC POWER FLOW EQUATIONS:

P_i = V_i * Σ (V_j * (G_ij * cos(θ_ij) + B_ij * sin(θ_ij)))
Q_i = V_i * Σ (V_j * (G_ij * sin(θ_ij) - B_ij * cos(θ_ij)))
Where:

P_i: Real power at bus i
Q_i: Reactive power at bus i
V_i: Voltage magnitude at bus i
θ_ij: Voltage angle difference between bus i and bus j
G_ij: Conductance of line ij
B_ij: Susceptance of line ij

"""

#columnsToConvert=[0,1,2]
#newLines=nw.lines
#newLines[:, columnsToConvert] = newLines[:, columnsToConvert].astype(int)
##print(newLines)


## BUT FIRST, LET'S CREATE THE ADMITTANCE Y MATRIX FOR THE SYSTEM
def calcAdmittance(R,X):
    return 1/(R+1j*X)

##print(calcAdmittance(0.3,0.5))

def createY(lines, nbus):
    Y = np.zeros((nbus,nbus), dtype=complex) # initialize Y matrix with 0
    
    for(lineNo, bus1, bus2, R, X, B,rating) in nw.lines:
        bus1 = int(bus1)-1
        bus2 = int(bus2)-1
        Y_line = 1/(R+1j*X) ##calcAdmittance(R, X)
        Y[bus1, bus2] -= Y_line
        Y[bus2, bus1] -= Y_line
        Y[bus1, bus1] += Y_line
        Y[bus2, bus2] += Y_line

    return Y

Y=createY(nw.lines, nbus)
print(f"Y {Y.shape} ADMITTANCE MATRIX IS : ")
print(Y)

## Active Power Vector P, Reactive Power Vector Q, Voltage vector V

P, Q = np.zeros((2,nbus), dtype = float)
V = np.zeros((nbus) , dtype = float)
theta = np.zeros((nbus), dtype = float)
Vcomplex = np.zeros((nbus), dtype = complex)
for(busNo, Vmag, Vangle, Pg, Qg, Pl, Ql, Qmin, Qmax) in nw.buses:
    P[int(busNo)-1] = Pg + Pl
    Q[int(busNo)-1] = Qg + Ql
    V[int(busNo)-1] = Vmag
    theta[int(busNo)-1] = np.radians(Vangle)
    Vcomplex[int(nbus)-1] = complex(Vmag*np.cos(np.radians(Vangle)), Vmag*np.sin(np.radians(Vangle)))

print(f"\n\n Active power P vector: \n {P} \n\n Reactive Power Q vector: \n {Q} \n\n Voltage magnitude V vector: \n {V} \n\n Voltage Angle Theta vector: \n {theta}")

## LET'S CREATE A FUNCTION TO CREATE THE JACOBIAN MATRIX
def JacobianMatrix(Y,V, theta):
    n = len(V)
    J = np.zeros((2 * n, 2 * n))
    
    # Extract real and imaginary parts of the admittance matrix Y first
    G = Y.real
    B = Y.imag

    # Compute common terms for computational efficiency of the program
    V_i = V[:, np.newaxis]  # Shape (n, 1)
    V_j = V[np.newaxis, :]  # Shape (1, n)
    theta_ij = theta[:, np.newaxis] - theta[np.newaxis, :]  # Shape (n, n)

    cos_theta = np.cos(theta_ij)
    sin_theta = np.sin(theta_ij)
    
    ## repeated terms definition for avoiding repetition
    G_cos = G * cos_theta
    B_sin = B * sin_theta
    G_sin = G * sin_theta
    B_cos = B * cos_theta

    # J11: Partial derivatives of P with respect to theta
    J[:n, :n] = -V_i * V_j * (G_sin - B_cos)
    
    # J12: Partial derivatives of P with respect to V
    J[:n, n:] = V_i * (G_cos + B_sin)

    # J21: Partial derivatives of Q with respect to theta
    J[n:, :n] = V_i * V_j * (G_cos + B_sin)
    
    # J22: Partial derivatives of Q with respect to V
    J[n:, n:] = -V_i * (G_sin - B_cos)

    return J

## NOW LET'S PERFORM POWER FLOW ANALYSIS USING NEWTON-RAPHSON ITERATIONS
def pfNewtonRaphson(Y, P, Q, V, theta, Vcomplex, tolerance = 1e-6, maxIteration = 4):
    ## tolerance: Convergence tolerance for iterations
    ## maxIteration: Maximum number of iterations
    
    nbus = len(V)
    for iteration in range(maxIteration):
    
        print(f"\n ITERATION : {iteration} \n")
        ## LET'S CALCULATE Pcalc and Q calc as power mismatches in each bus
        Pcalc = np.zeros(nbus, dtype = float)
        Qcalc = np.zeros(nbus, dtype = float)

        for i in range(nbus):
            Pcalc[i] = np.real(Vcomplex[i] * np.sum(Vcomplex * np.conj(Y[i, :])))
            Qcalc[i] = np.imag(Vcomplex[i] * np.sum(Vcomplex * np.conj(Y[i, :])))

        Pdiff = P - Pcalc ##Calculate the difference between actual value and the calculated value
        Qdiff = Q - Qcalc

        if np.all(np.abs(Pdiff) < tolerance) and np.all(np.abs(Qdiff) < tolerance): ##Condition to test the convergence
            return Vcomplex
        

        """
        ## NOW LET'S CREATE THE JACOBIAN MATRIX FOR CALCULATION
        J = np.zeros((2*nbus, 2*nbus), dtype = complex)
        for i in range(nbus):
            for j in range(nbus):
                if i != j:
                    ## Let's calculate angle and magnitude differences
                    thetaDelta = np.angle(V[i]) - np.angle(V[j])
                    sinDelta = np.sin(thetaDelta)
                    cosDelta = np.cos(thetaDelta)

                    ## Compute partial derivatives
                    
                    J[i, j] = -V[i] * np.conj(V[j]) * Y[i, j]
                    J[i+nbus, j] = -V[i] * np.conj(V[j]) * (np.real(Y[i, j]) * sinDelta - np.imag(Y[i, j]) * cosDelta)
                    J[i, j+nbus] = V[i] * np.conj(V[j]) * (np.imag(Y[i, j]) * cosDelta - np.real(Y[i, j]) * sinDelta)
                    J[i+nbus, j+nbus] = V[i] * np.conj(V[j]) * (-np.imag(Y[i, j]) * sinDelta - np.real(Y[i, j]) * cosDelta)
               
        """

        J = JacobianMatrix(Y, V, theta)
        print(f"\n The Jacobian Matrix J is as:\n{J}")
        # LET'S CALCULATE UPDATED VALUES USING LINEAR SOLVERS
        dX = np.linalg.solve(J, np.concatenate([Pdiff, Qdiff]))
        
        # NEW VOLTAGE VALUES
        Vcomplex.real += dX[:nbus]
        Vcomplex.imag += dX[nbus:]
    
    raise ValueError(f"Newton-Raphson Iteration did not converge after {iteration+1} iterations.")

# Solve power flow
V = pfNewtonRaphson(Y, P, Q, V, theta, Vcomplex)

# Print results
print("Voltage Magnitudes and Angles:")
for i in range(len(V)):
    print(f"Bus {i+1}: Magnitude = {abs(V[i]):.2f}, Angle = {np.angle(V[i], deg=True):.2f} degrees")
