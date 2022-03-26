function [wx] = Vecteur_composant_principale(l,X,U)
% pour calculer le vecteur de ses ` composantes principales
for i = 1:l
    wx(i) = X' * U(:,i);
end 
end

