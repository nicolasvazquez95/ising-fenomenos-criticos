using Random

function ising2d!(s, J, h, T, niters,seed=42)
    rng = Xoshiro(seed)
    m, n = size(s)
    N = m*n
    # Max and min values of the sum of spins in the neighborhood
    min_k = -4
    max_k = 4
    β = 1/T # Inverse temperature (k_B = 1)
    # Glauber dynamics
    probs = [ 1 / (1 + exp(-2*β*(k * J + h))) for k in min_k:max_k]

    for mcs in 1:niters
        for _ in 1:N
            idx = rand(rng, 1:N)
            # divrem returns the quotient and remainder of integer division
            j0, i0 = divrem(idx - 1, m)
            i = i0 + 1
            j = j0 + 1
            # NN is the up spin in the neighborhood of S[i,j]
            NN = s[ifelse(i==1, m, i-1), j]
            # SS is the down spin in the neighborhood of S[i,j]
            SS = s[ifelse(i==m, 1, i+1), j]
            # WW is the left spin in the neighborhood of S[i,j]
            WW = s[i, ifelse(j==1, n, j-1)]
            # EE is the right spin in the neighborhood of S[i,j]
            EE = s[i, ifelse(j==n, 1, j+1)]
            k = NN + SS + WW + EE
            # Flip the spin with probability p
            s[i,j] = ifelse(rand(rng) < probs[k - min_k + 1], +1, -1)
        end
    end
end



