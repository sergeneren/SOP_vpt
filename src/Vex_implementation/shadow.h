int shadow(vector orig; vector L) {
	vector pos_shadow;
	vector dir = L - orig;
	float u, v;
	int shadow = !(intersect(1, orig + (dir * 0.01), dir, pos_shadow, u, v)>0);
	return shadow;
}

float vol_shadow(vector orig; vector L) {
	vector pos_shadow = orig;
	vector dir = normalize(L - orig);
	float u, v;

	float density = 0; 
	float offset = 0; 
	float l_len = length(L - pos_shadow);
	
	while (offset < l_len) {
		
		if (density > 1) break; 
		pos_shadow += dir * offset;
		density += volumesample(2, "density", pos_shadow);
		offset += 0.1; 
		
	}
	
	return density;
}