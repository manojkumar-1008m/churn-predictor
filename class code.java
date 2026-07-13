import java.util.Scanner;
class Strong {
    boolean check(int n) {
        int sum = 0, temp = n;
        while(temp > 0) {
            int rem = temp % 10;
            int fact = 1;
            for(int i = 1; i <= rem; i++) {
                fact = fact * i;
            }
            sum = sum + fact;
            temp = temp / 10;
        }

        return sum == n;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        Strong obj = new Strong();

        if(obj.check(n))
            System.out.println("Strong Number");
        else
            System.out.println("Not Strong Number");
    }
}