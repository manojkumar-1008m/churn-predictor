import java.util.Scanner;
class ArmstrongNumber {
    boolean check(int n) {
        int sum = 0, temp = n;
        while(temp > 0) {
            int rem = temp % 10;
            sum = sum + (rem * rem * rem);
            temp = temp / 10;
        }

        return sum == n;
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        ArmstrongNumber obj = new ArmstrongNumber();

        if(obj.check(n))
            System.out.println("Armstrong Number");
        else
            System.out.println("Not Armstrong Number");
    }
}


