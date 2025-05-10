import numpy as np


from scipy.sparse import diags

#from functools import cached_property  #python >3.8
from functools import lru_cache


class CochlearModel:
	def __init__(self, H=1e-3, L=3.2e-2, sig_bm=0.06, rho=1000, H_TM=0.025e-3, mu=150e-3, r=0.25, 
		apply_mult_factor_RL=True):
		'''
		Args:
			H_TM: height of TM
			mu: shear viscosity constant TM (Pa.s)
			r: ratio TM width/BM width
			apply_mult_factor_RL: if True, apply mult factor for v_RL/v_BM 
		'''

		self._H = H 
		self.L = L 
		self._sig_bm = sig_bm
		self.rho = rho
		self.H_TM=H_TM 
		self.mu=mu
		self.r = r

		self.apply_mult_factor_RL=apply_mult_factor_RL
		self._mult_factor=1. #init

		self.scaling_symmetry=False

		self.omega_bm_0=2*np.pi*(10000)

	def activate_scaling_symmetry(self, flag=True):
		self.scaling_symmetry=flag

	def H(self, omega_bm):
		if self.scaling_symmetry:
			return self._H*omega_bm/self.omega_bm_0
		else:
			return self._H

	def sig_bm(self, omega_bm):
		if self.scaling_symmetry:
			return self._sig_bm/omega_bm*self.omega_bm_0
		else:
			return self._sig_bm

	

	def alpha(self, k, omega_bm, eps=1e-6):
		return (k*self.H(omega_bm)+eps)/np.tanh(k*self.H(omega_bm)+eps)

	def k_it(self, omega_bm, a, G, k, omega, funcVisc=None): 
		'''
		Performs an iteration of the recursive procedure Parameters a and funcVisc provide two ways to control viscous stress.
		Args:
			a: typically between 0 and 1. Controls the strength of viscous force (0: mu=0, 1: mu=mu_0)
			funcVisc: function of x=omega/omega_bm to modify viscous forces (multiplicative factor of visc is : 1-func )
		'''
		alpha_=self.alpha(k, omega_bm)
		

		#DEBUG (alternative way to compute k)
		#visc=self.compute_visc(omega_bm, a, G, k, omega, funcVisc=funcVisc)
		#gamma=(1-G)*omega_bm
		
		#num= 2*alpha_*omega**2*self.rho
		#den= self.sig_bm*self.H*(omega_bm**2-omega**2+
	#		1j*omega*(gamma+visc/self.sig_bm))
		
		
		Y_BM=self.compute_YBM(omega_bm, a, G, k, omega, funcVisc=funcVisc)
		z=2*alpha_*1j*self.rho*omega/self.H(omega_bm)*Y_BM
		#z=num/den


		z_mod=np.abs(z)
		z_phase=np.unwrap(np.angle(z))

		return np.sqrt(z_mod)*np.exp(1j*z_phase/2.)

		#return np.sqrt(num/den)

	def k_init(self, omega_bm, G, omega):
		gamma=(1-G)*omega_bm
		alpha_=1.
		
		gamma=(1-G)*omega_bm
		visc=0

		num= 2*alpha_*omega**2*self.rho

		den= self.sig_bm(omega_bm)*self.H(omega_bm)*(omega_bm**2-omega**2+
			1j*omega*(gamma+visc/self.sig_bm(omega_bm)))
		
		z=num/den
		z_mod=np.abs(z)
		z_phase=np.unwrap(np.angle(z))

		return np.sqrt(z_mod)*np.exp(1j*z_phase/2.)

	def compute_YBM(self, omega_bm, a, G, k, omega, funcVisc=None):
		visc=self.compute_visc(omega_bm, a, G, k, omega, funcVisc=funcVisc)
		gamma=(1-G)*omega_bm

		num= -1j*omega
		den= self.sig_bm(omega_bm)*(omega_bm**2-omega**2+
			1j*omega*(gamma+visc/self.sig_bm(omega_bm)))
		return num/den


	def set_constant_mult_factor(self, mult_factor):
		self._mult_factor= mult_factor

	def compute_visc(self, omega_bm, a, G, k, omega, funcVisc=None):
		'''computes damping factor due to viscosity in TM '''

		k_z=k/np.sqrt(2)

		diff_term=1-np.exp(-k_z*self.H_TM)

		visc=2*a*self.r*self.mu*diff_term*k_z

		if self.apply_mult_factor_RL:
			visc*=self._mult_factor


		if not funcVisc is None:
			x=omega/omega_bm
			visc*=1-funcVisc(x)		

		#HACK add effect viscous fluid from water
		#b=2.5
		#alpha_=self.alpha(k, omega_bm)
		#mu_w=1e-3   
		#mu_w=10e-3 #hack 10 mu_w (cf Sisto paper) 
		#mu_w=5e-3  #similar behavior to viscous load for mu_tm=150 mpas
		#visc+=4*b*mu_w*alpha_/self.H(omega_bm)

		return visc


	def createCochlPart(self, n_sections=1000, A=4.2e5, B=-138.):
		'''Useful for WKB approx/finite differences
		omega_bm= Aexp(Bx) '''
		self.n_sections=n_sections
		self.x=np.linspace(0, self.L, num=n_sections)
		self.omega_bm_x=A*np.exp(B*self.x)
		self.A=A
		self.B=B



	def compute_x(self, f_bm):
		return 1./self.B*np.log(2*np.pi*f_bm/self.A)

	def compute_f(self, x):
		return self.A*np.exp(self.B*x)/(2*np.pi)


	def compute_FD_matrix(self, k):
		'''Computes finite difference matrix (F)
		Note: k is computed outside function to let the user decide how it is calculated (also note that k!=2pi/lambda in general). Must be of size n_sections'''

		n=self.n_sections

		dx=self.L/n
		dx2=dx**2
		main_diag = k**2*dx2-2
		main_diag[0]= dx2 
		main_diag[-1]=dx2
		upper_diag = np.ones(n-1)
		lower_diag = np.ones(n-1)
		upper_diag[0]=0
		lower_diag[n-2]=0
		tridiag_matrix = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], shape=(n, n))
		tridiag_matrix=tridiag_matrix.tocsr()
		return 1/dx2*tridiag_matrix
