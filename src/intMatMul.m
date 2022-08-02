function [lower, upper] = intMatMul(A1, A2, B1, B2)
    A_center = (A1 + A2)/2;
    A_radius = (A2 - A1)/2;

    B_center = (B1 + B2)/2;
    B_radius = (B2 - B1)/2;

    C = intervalMatrix(A_center, A_radius);
    D = intervalMatrix(B_center, B_radius);

    result = C * D;

    lower = infimum(result.int);
    upper = supremum(result.int);
end
