 clear all
 close all
 
 w = zeros(1, 16);
 err = 1;
 while err > 0.001
     v = T(w);
     err = max(abs(v - w));
     w = v;
     hold on;
     plot(w);
 end
 hold off;
