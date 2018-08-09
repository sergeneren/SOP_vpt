int shadow(vector orig; vector L) {
	vector pos_shadow;
	vector dir = L - orig;
	float u, v;
	int shadow = !(intersect(1, orig + (dir * 0.01), dir, pos_shadow, u, v)>0);
	return shadow;
}

