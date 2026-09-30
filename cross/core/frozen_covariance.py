"""Exact block storage for a diagonal, fixed Schmidt nuisance covariance.

V = [[A, B], [B.T, diag(d)]] in an explicit coordinate ordering. Only the
nuisance block is diagonal: every active correlation and active/nuisance
cross-covariance is retained. This is standard block Gaussian algebra, not a
bounded-map estimator. Memory is O(a*a + a*g + g), not O((a+g)**2).
"""
import numpy as np


class FrozenDiagonalCovariance:
    __array_priority__ = 10000
    format = 'fixed-diagonal-schmidt-v1'

    def __init__(self, size, active_indices, frozen_indices, active, cross, diagonal):
        self.shape = (int(size), int(size))
        self.active_indices = np.array(active_indices, dtype=np.intp, copy=True)
        self.frozen_indices = np.array(frozen_indices, dtype=np.intp, copy=True)
        self.active = np.array(active, dtype=float, copy=True)
        self.cross = np.array(cross, dtype=float, copy=True)
        self.frozen_diagonal = np.array(diagonal, dtype=float, copy=True)
        a, g = len(self.active_indices), len(self.frozen_indices)
        if (self.active_indices.ndim != 1 or self.frozen_indices.ndim != 1
                or self.active.shape != (a,a) or self.cross.shape != (a,g)
                or self.frozen_diagonal.shape != (g,)
                or not np.array_equal(np.sort(np.r_[self.active_indices,self.frozen_indices]),np.arange(size))):
            raise ValueError('Structured covariance dimensions/partition are inconsistent')
        if not all(np.isfinite(x).all() for x in (self.active,self.cross,self.frozen_diagonal)):
            raise ValueError('Structured covariance must be finite')
        if not np.allclose(self.active,self.active.T,rtol=0,atol=1e-12):
            raise ValueError('Active source covariance must be symmetric')
        self.active=(self.active+self.active.T)/2
        if np.any(self.active.diagonal() < -1e-12) or np.any(self.frozen_diagonal <= 0):
            raise ValueError('Frozen variances must be positive; active variances nonnegative')

    @classmethod
    def from_dense(cls, covariance, frozen_indices):
        V = np.asarray(covariance,dtype=float)
        g = np.array(frozen_indices,dtype=np.intp)
        a = np.setdiff1d(np.arange(len(V)),g)
        G = V[np.ix_(g,g)]
        if not np.array_equal(G,np.diag(G.diagonal())):
            raise ValueError('Cannot approximate a correlated frozen block as diagonal')
        if not np.allclose(V,V.T,rtol=0,atol=1e-12):
            raise ValueError('Dense source covariance must be symmetric before conversion')
        cross=(V[np.ix_(a,g)]+V[np.ix_(g,a)].T)/2
        return cls(len(V),a,g,V[np.ix_(a,a)],cross,G.diagonal())

    def copy(self):
        return type(self)(self.shape[0],self.active_indices,self.frozen_indices,
            self.active,self.cross,self.frozen_diagonal)

    @property
    def T(self):
        return self

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.active_indices,self.frozen_indices,
            self.active,self.cross,self.frozen_diagonal))

    def diagonal(self):
        result=np.empty(self.shape[0])
        result[self.active_indices]=self.active.diagonal()
        result[self.frozen_indices]=self.frozen_diagonal
        return result

    def __matmul__(self, other):
        x=np.asarray(other,dtype=float)
        vector=x.ndim==1
        if vector:x=x[:,None]
        if x.ndim!=2 or x.shape[0]!=self.shape[0]:
            raise ValueError('Source covariance multiplication requires compatible columns')
        a,g=x[self.active_indices],x[self.frozen_indices]
        out=np.empty_like(x)
        out[self.active_indices]=self.active@a+self.cross@g
        out[self.frozen_indices]=self.cross.T@a+self.frozen_diagonal[:,None]*g
        return out[:,0] if vector else out

    def __rmatmul__(self, other):
        return (self@np.asarray(other,dtype=float).T).T

    def __array__(self, dtype=None, copy=None):
        raise TypeError('Implicit dense allocation is forbidden; use to_dense only for explicit diagnostics')

    def to_dense(self):
        """Explicit compatibility/diagnostic export, never called by filtering."""
        result=np.zeros(self.shape)
        a,g=self.active_indices,self.frozen_indices
        result[np.ix_(a,a)]=self.active
        result[np.ix_(a,g)]=self.cross
        result[np.ix_(g,a)]=self.cross.T
        result[g,g]=self.frozen_diagonal
        return result

    def append(self, variances, frozen):
        """Add independent coordinates, retaining the old full source order."""
        variances=np.asarray(variances,dtype=float);frozen=np.asarray(frozen,dtype=bool)
        if (variances.ndim!=1 or frozen.shape!=variances.shape
                or not np.isfinite(variances).all() or np.any(variances<0)):
            raise ValueError('Invalid appended source priors')
        a,g=len(self.active_indices),len(self.frozen_indices)
        new=np.arange(self.shape[0],self.shape[0]+len(variances))
        na,ng=new[~frozen],new[frozen]
        active=np.zeros((a+len(na),a+len(na)));active[:a,:a]=self.active
        i=np.arange(a,a+len(na));active[i,i]=variances[~frozen]
        cross=np.zeros((a+len(na),g+len(ng)));cross[:a,:g]=self.cross
        return type(self)(self.shape[0]+len(variances),np.r_[self.active_indices,na],
            np.r_[self.frozen_indices,ng],active,cross,np.r_[self.frozen_diagonal,variances[frozen]])

    def schmidt_update(self, gain, source_innovation_cross, innovation):
        """Full Joseph source update with exactly zero nuisance gain."""
        if np.any(gain[self.frozen_indices]!=0):
            raise ValueError('A fixed nuisance block cannot receive an active gain')
        k=gain[self.active_indices];u=source_innovation_cross[self.active_indices]
        active=self.active-k@u.T-u@k.T+k@innovation@k.T
        cross=self.cross-k@source_innovation_cross[self.frozen_indices].T
        return type(self)(self.shape[0],self.active_indices,self.frozen_indices,
            (active+active.T)/2,cross,self.frozen_diagonal)

    def conditional_response(self, pose_cross, dense_response):
        """Schur solve, including singular active support without fake jitter."""
        a,g=self.active_indices,self.frozen_indices
        scaled=self.cross/self.frozen_diagonal
        schur=self.active-scaled@self.cross.T
        schur=(schur+schur.T)/2
        rhs=pose_cross[:,a]-pose_cross[:,g]@scaled.T
        response=np.zeros_like(pose_cross)
        response[:,a]=dense_response(schur,rhs)
        response[:,g]=(pose_cross[:,g]-response[:,a]@self.cross)/self.frozen_diagonal
        if not np.allclose(response@self,pose_cross,rtol=1e-8,atol=1e-10):
            raise ValueError('Pose/source cross-covariance lies outside structured support')
        return response

    def validate_psd(self):
        # The diagonal nuisance block is strictly positive. The full matrix is
        # PSD exactly when its active Schur complement is PSD.
        schur=self.active-(self.cross/self.frozen_diagonal)@self.cross.T
        if len(schur) and np.linalg.eigvalsh((schur+schur.T)/2).min() < -1e-10:
            raise ValueError('Saved structured source covariance is not positive semidefinite')

    def record(self):
        return dict(format=self.format,size=self.shape[0],active_indices=self.active_indices.tolist(),
            frozen_indices=self.frozen_indices.tolist(),active=self.active.tolist(),cross=self.cross.tolist(),
            frozen_diagonal=self.frozen_diagonal.tolist())

    @classmethod
    def from_record(cls, record):
        if record.get('format')!=cls.format:
            raise ValueError('Unsupported structured source covariance format')
        a,g=len(record['active_indices']),len(record['frozen_indices'])
        result=cls(record['size'],record['active_indices'],record['frozen_indices'],
            np.asarray(record['active']).reshape(a,a),np.asarray(record['cross']).reshape(a,g),
            record['frozen_diagonal'])
        result.validate_psd()
        return result
