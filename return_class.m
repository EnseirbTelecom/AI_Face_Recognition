function [class, class_decide] = return_class(k,Vx_sort,Vx,lines,cls_trn)
%RETURN_CLASS 
for i = 1:k
    t = Vx_sort(i);
    idx = find(Vx == t);
    class(i) = cls_trn(ceil(idx/lines));
end
class_decide = mode(class);

end

