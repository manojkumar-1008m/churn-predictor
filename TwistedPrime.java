import java.util.Scanner;

class TwistedPrime {//WITH PARAMETER WITH RETURN

    boolean check(int n) {

        int rev = 0, temp = n;

        while(temp > 0) {
            rev = rev * 10 + temp % 10;
            temp = temp / 10;
        }
        int c1 = 0, c2 = 0;
        for(int i = 1; i <= n; i++) {
            if(n % i == 0)
                c1++;
        }

        for(int i = 1; i <= rev; i++) {
            if(rev % i == 0)
                c2++;
        }

        return c1 == 2 && c2 == 2;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        TwistedPrime obj = new TwistedPrime();

        if(obj.check(n))
            System.out.println("Twisted Prime");
        else
            System.out.println("Not Twisted Prime");
    }
}