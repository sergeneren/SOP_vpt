struct rayData {
	float offsets[];
	vector sigma_s[];
	vector sigma_t[];
	vector T[];
	float cdfs[];
	float pdfs[];


	// return values for given offset
	void getValuesFromOffset(float offset, pdf; vector _sigma_s, _T)
	{
		int i = 0;
		// TODO binary search
		while (offsets[i + 1] < offset  && i<len(offsets)) i++;
		float k = (offset - offsets[i]) / (offsets[i + 1] - offsets[i]);
		pdf = lerp(pdfs[i], pdfs[i + 1], k);
		_sigma_s = lerp(sigma_s[i], sigma_s[i + 1], k);
		vector _sigma_t = lerp(sigma_t[i], sigma_t[i + 1], k);
		_T = T[i] * exp(-(offset - offsets[i]) * _sigma_t);
	}

	// return values for given random value
	void getValuesFromRandom(float r, pdf, offset; vector _sigma_s, _T)
	{
		int i = 0;
		r = lerp(cdfs[0], cdfs[-1], r);
		// TODO binary search
		while (cdfs[i + 1] < r) i++;
		float k = (r - cdfs[i]) / (cdfs[i + 1] - cdfs[i]);

		pdf = lerp(pdfs[i], pdfs[i + 1], k) / cdfs[-1] / (offsets[i + 1] - offsets[i]);
		// TODO exponential interpolation
		offset = lerp(offsets[i], offsets[i + 1], k);
		_sigma_s = lerp(sigma_s[i], sigma_s[i + 1], k);
		vector _sigma_t = lerp(sigma_t[i], sigma_t[i + 1], k);
		_T = T[i] * exp(-(offset - offsets[i]) * _sigma_t);
	}
}


void compute(vector cam, L, dir, outColor, opacity , scattering, absorption;
			float maxdist, sh_density, step_rate;
			int indirect_samples, volume_index) {

	vector extinction = absorption + scattering;
	float ext_lum = luminance(extinction);
	float weight = 1.0 / indirect_samples;
	float dist = min(maxdist, length(L));
	vector nI = normalize(dir);
	
	
	
	for (int i = 0; i < indirect_samples; i++) {
	
		vector clr;
		vector distrib = 0;
		
		float delta = dot(L, nI);
		
		float D = length(delta*nI - L);
	
		vector p[];
		int prim[]; 
		vector uvw[]; 

		int intersection = intersect_all(3, cam, dir*maxdist, p, prim, uvw, 0 , 0); 
		if (intersection < 0) return; 
		
		float ray_len = length(p[1] - p[0]); 
		
		rayData data; 
		float offset = 0; 
		
		vector transmittance = 1;
		float density = 0;
		float sum_pdfs = 0;
		float pdf;


		
		float diagonal = volumevoxeldiameter(volume_index, "density") * 0.577350269;// divide by sqrt(3)
		float step_size = diagonal / step_rate;

		
		while (offset < ray_len && (max(transmittance) > 0.0001)) {
			vector pos = p[0] + dir * offset; 
			density = volumesample(volume_index, "density", pos); 
			transmittance *= exp(-density * extinction * step_size); 
			float shad = vol_shadow(pos, L); 

			if(density>0) outColor = (1 - transmittance) * (1 - shad); 
			
			pdf = luminance(transmittance * density * scattering);
			sum_pdfs += pdf;
		
			push(data.offsets, offset);
			push(data.T, transmittance);
			push(data.sigma_s, density * scattering);
			push(data.sigma_t, density * extinction);
			push(data.pdfs, pdf);
			push(data.cdfs, sum_pdfs);

			offset += step_size;
		}

		if (len(data.T)<2) return;
		if (transmittance == 1) return;
		






	}

}

